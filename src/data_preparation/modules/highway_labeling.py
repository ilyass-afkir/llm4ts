"""Labeling GPS trajectory points with highway types using OpenStreetMap (OSM).

This module loads verified trip datasets, assigns highway labels to each 
GPS point using spatial matching against offline or online OSM sources,
and writes the labeled outputs and verification plots to disk.

Example:
    >>> from omegaconf import OmegaConf
    >>> cfg = OmegaConf.load("configs/data_preparation.yaml")
    >>> labeler = TruckDataHighwayLabeler(cfg)
    >>> labeler.run()
"""

import logging
from pathlib import Path
from typing import Tuple, List, Dict
import time
import json
import gc

import requests
import numpy as np
import pandas as pd
from omegaconf import DictConfig  
from tqdm import tqdm
import geopandas as gpd
from osmnx.features import features_from_bbox
from osmnx.graph import graph_from_bbox 
from osmnx.distance import nearest_edges
import folium
from shapely.geometry import LineString
from pyrosm import OSM
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

from src.data_preparation.modules import constants as const 

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")

class TruckDataHighwayLabeler:
    """Labels GPS trajectory points of electric trucks with OSM highway types.
    
    Loads verified electric truck trip datasets, spatially matches each GPS
    point against offline or online OpenStreetMap road networks, and assigns
    the appropriate highway type label (e.g. motorway, trunk, residential).
    Labeled outputs are saved to disk together with interactive Folium
    verification maps.

    Attributes:
        cfg (DictConfig): Hydra configuration object.
        verified_data_dir (Path): Directory containing verified parquet trip files.
        file_extension (str): File suffix used when globbing trip files (e.g. ``'.parquet'``).
        labeled_data_dir (Path): Directory where labeled parquet files are written.
        plots_dir (Path): Directory where Folium HTML verification maps are written.
        trip_locations_file (Path): Path to the JSON file mapping country keys to trip filenames
            created by :meth:`data_understanding.modules.analysis.TruckDataAnalyzer.get_trip_locations`.
        osm_maps_dir (Path): Root directory containing the offline OSM PBF maps.
        processed_osm_maps_dir (Path): Cache directory for processed highway GeoDataFrames.
        highway_labeling_signals (list of str): Signals used for highway labeling.
        highway_labels (list of str): List of OSM highway labels.
        truck_speed_ranges (dict): Recommended speed ranges per highway type.
        highway_priority (dict): Priority ranking for conflicting highway assignments.
        country_code_to_osm_map (dict): Mapping from country codes to OSM PBF filenames.
        crs_codes (dict): CRS projection codes per country.
    """
    def __init__(self, cfg: DictConfig):  
        """Initializes the labeler from a Hydra configuration object.

        Args:
            cfg (DictConfig): Hydra configuration object. Must contain a
                ``data_preparation`` sub-config with the fields used by
                this class.
        """
        self.cfg = cfg
        self.verified_data_dir = Path(self.cfg.data_preparation.verified_data_dir)
        self.file_extension = self.cfg.data_preparation.file_extension
        self.labeled_data_dir = Path(self.cfg.data_preparation.labeled_data_dir)
        self.plots_dir = Path(self.cfg.data_preparation.plots_dir)
        self.trip_locations_file = Path(self.cfg.data_preparation.trip_locations_file)
        self.osm_maps_dir = Path(self.cfg.data_preparation.osm_maps_dir)
        self.processed_osm_maps_dir = Path(self.cfg.data_preparation.processed_osm_maps_dir)
        self.highway_labeling_signals = const.HIGHWAY_LABELING_SIGNALS
        self.highway_labels = const.HIGHWAY_LABELS
        self.truck_speed_ranges = const.TRUCK_SPEED_RANGES
        self.highway_priority = const.HIGHWAY_PRIORITY
        self.country_code_to_osm_map = const.COUNTRY_CODE_TO_OSM_MAP
        self.crs_codes = const.CRS_CODES
        self.setup_dirs()
  
    def setup_dirs(self):
        """Creates required output directories if they do not already exist."""
        for path in [self.processed_osm_maps_dir, self.labeled_data_dir, self.plots_dir]:
            path.mkdir(parents=True, exist_ok=True)

    def get_datasets(self) -> Tuple[List[Path], List[str]]:
        """Gets all trip files in the verified data directory.

        Returns:
            Tuple[List[Path], List[str]]: A tuple of ``(paths, filenames)`` where
                ``paths`` is a sorted list of :class:`pathlib.Path` objects
                matching ``file_extension``, and ``filenames`` contains the
                corresponding stems (no extension).
        """
        paths = sorted(self.verified_data_dir.glob(f"*{self.file_extension}"))
        filenames = [f.stem for f in paths]
        return paths, filenames

    def get_datasets_by_country(self) -> Dict[str, Tuple[List[Path], List[str]]]:
        """Groups trip file paths by country (or country combination) key.

        Reads the JSON file at ``trip_locations_path`` which maps each country
        key to a list of trip filenames, then resolves absolute paths for every
        filename.

        Returns:
            Dict[str, Tuple[List[Path], List[str]]]: A dict mapping each country 
            key (e.g. ``"DE"`` or ``"DK,SE"``) to a tuple ``(paths, filenames)`` of 
            resolved :class:`~pathlib.Path` objects and their stems.
        """
        with open(self.trip_locations_file, "r") as f:
            trip_locations = json.load(f)

        country_dict = {
            key: (
                [self.verified_data_dir / f"{fn}{self.file_extension}" for fn in filenames],
                filenames
            )
            for key, filenames in trip_locations.items()
        }

        total_files = sum(len(filenames) for _, filenames in country_dict.values())
        logger.info("Total number of files across all keys: %d", total_files)

        return country_dict

    def plot_and_verify_labels(self, df: pd.DataFrame, filename: str, country: str) -> None:
        """Renders an interactive Folium map visualising the highway labels for one trip.

        Each trajectory segment is coloured by its assigned highway type. A
        legend shows the distribution of labels as percentages. The map is written
        to disk as an HTML file.

        Args:
            df (pd.DataFrame): Labeled trip DataFrame. Must contain
                ``latitude_cval_ippc``, ``longitude_cval_ippc``, and
                ``highway_label`` columns.
            filename (str): Stem of the source file, which is used as the map title
                and output filename.
            country (str): Country key used to create a country-level
                sub-directory under ``plots_dir``.
                
        Raises:
            ValueError: If ``df`` is empty.

        Note:
            This function was developed with the assistance of Claude AI (Anthropic).
        """
        normal_font_file = "llm-erange/src/utils/times.ttf"
        bold_font_file  = "llm-erange/src/utils/times_bold.ttf"
        fm.fontManager.addfont(normal_font_file)
        fm.fontManager.addfont(bold_font_file)
        prop_normal = fm.FontProperties(fname=normal_font_file)
        plt.rcParams["font.family"] = prop_normal.get_name()
        plt.rcParams["font.size"] = 12
            
        if df.empty:
            raise ValueError("DataFrame is empty. Cannot plot trajectory.")
        
        m = folium.Map(
            location=[df["latitude_cval_ippc"].mean(), df["longitude_cval_ippc"].mean()],
            zoom_start=12
        )
        
        highway_colors = {
            'motorway': '#FF0000',        
            'motorway_link': '#CC0000',   
            'trunk': '#FF6600',           
            'trunk_link': '#FF4500',      
            'primary': '#FFD700',         
            'primary_link': '#FFA500',    
            'secondary': '#00FF00',       
            'secondary_link': '#00CC00',  
            'tertiary': '#00BFFF',        
            'tertiary_link': '#1E90FF',   
            'residential': '#FF00FF',     
            'living_street': '#DA70D6',   
            'service': '#00FFFF',         
            'track': '#8B4513',           
            'unclassified': '#808080',    
            'unlabeled': '#D3D3D3'        
        }

        # Draw trajectory as coloured segments (one PolyLine per GPS point pair)
        for i in range(len(df) - 1):
            segment = [
                (df.iloc[i]["latitude_cval_ippc"], df.iloc[i]["longitude_cval_ippc"]),
                (df.iloc[i+1]["latitude_cval_ippc"], df.iloc[i+1]["longitude_cval_ippc"])
            ]
            
            highway_label = df.iloc[i]["highway_label"]
            
            if pd.isna(highway_label):
                color = highway_colors['unlabeled']
                label_text = "unlabeled"
            else:
                color = highway_colors.get(highway_label, '#000000')
                label_text = highway_label
            
            folium.PolyLine(
                segment,
                color=color,
                weight=4,
                opacity=0.8,
                tooltip=label_text
            ).add_to(m)
        
        # Start and end markers
        folium.Marker(
            (df.iloc[0]["latitude_cval_ippc"], df.iloc[0]["longitude_cval_ippc"]),
            tooltip="Start",
            icon=folium.Icon(color="green", icon="play")
        ).add_to(m)
        folium.Marker(
            (df.iloc[-1]["latitude_cval_ippc"], df.iloc[-1]["longitude_cval_ippc"]),
            tooltip="End",
            icon=folium.Icon(color="red", icon="stop")
        ).add_to(m)

        # Compute label distribution for legend
        total = len(df) - 1  
        segment_labels = df["highway_label"].iloc[:-1] 
        n_unlabeled = segment_labels.isna().sum()
        highway_counts = segment_labels.value_counts(dropna=True)
        highway_percentages = (highway_counts / total * 100).round(3)
        
        # Append unlabeled row only if unlabeled segments exist
        if n_unlabeled > 0:
            unlabeled_percentage = (n_unlabeled / total * 100).round(3)
        
        legend_html = f'''
            <div style="
                position:         fixed;
                top:              50px;
                right:            50px;
                background-color: white;
                border:           2px solid #333;
                z-index:          9999;
                font-family:      'Times New Roman';
                font-size:        16px;
                padding:          15px;
                border-radius:    5px;
                box-shadow:       2px 2px 6px rgba(0,0,0,0.3);
                width:            auto;
                min-width:        300px;">

                <p style="
                    margin:        0 0 12px 0;
                    border-bottom: 2px solid #333;
                    padding-bottom: 8px;
                    font-family:   'Times New Roman';
                    font-size:     16px;">
                    <strong>{filename}</strong> | 🟢 Start | 🔴 End
                </p>

                <p style="margin: 8px 0; font-size: 16px;">
                    Total segments: {total}
                </p>

                <hr style="
                    margin:      10px 0;
                    border:      none;
                    border-top:  1.3px solid #999;">
        '''

        for highway_label, percentage in highway_percentages.items():
            color = highway_colors.get(highway_label, '#000000')
            legend_html += f'''
            <p style="margin: 3px 0; font-size: 16px;">
                <span style="display: inline-block; 
                            width: 22px; 
                            height: 3px; 
                            background-color: {color}; 
                            margin-right: 8px; 
                            vertical-align: middle;"></span>
                {highway_label}: {percentage}%
            </p>
            '''

        if n_unlabeled > 0:
            color = highway_colors['unlabeled']
            legend_html += f'''
            <p style="margin: 3px 0; font-size: 16px;">
                <span style="display: inline-block; 
                            width: 22px; 
                            height: 3px; 
                            background-color: {color}; 
                            margin-right: 8px; 
                            vertical-align: middle;"></span>
                unlabeled: {unlabeled_percentage}%
            </p>
            '''

        legend_html += '''
        </div>
        '''
        m.get_root().html.add_child(folium.Element(legend_html))
        
        # Save map to country-level subdirectory
        save_country_plot_dir = self.plots_dir / country 
        save_country_plot_dir.mkdir(parents=True, exist_ok=True)
        m.save(save_country_plot_dir / f"{filename}.html")   

    def calculate_weighted_score(self, gdf_labeled: gpd.GeoDataFrame, max_distance: int) -> pd.Series:
        """Computes a weighted score for each candidate highway match.

        The score combines three normalized sub-scores:

        * **Speed score** (weight 0.3): ``1.0`` if the point's vehicle speed falls within the 
            expected range for the candidate highway type, else ``0.0``.
        * **Priority score** (weight 0.3): highway priority value normalized to 
            ``[0, 1]`` using :attr:`data_preparation.modules.constants.HIGHWAY_PRIORITY`.
        * **Distance score** (weight 0.4): linear decay from ``1.0`` (distance = 0) 
            to ``0.0`` (distance = ``max_distance``).

        Args:
            gdf_labeled (gpd.GeoDataFrame): Candidate matches from a spatial join.
                Must contain ``vehspd_cval_cpc``, ``highway``, and
                ``dist_to_highway_m`` columns.
            max_distance (int): Maximum search radius in metres used during the
                spatial join; serves as the normalisation denominator for the
                distance score.

        Returns:
            pd.Series: Float scores in ``[0, 1]`` aligned to ``gdf_labeled``'s index.
        """
        speed_col = gdf_labeled['vehspd_cval_cpc'].values
        hw_col = gdf_labeled['highway'].values
        speed_scores = np.zeros(len(gdf_labeled))
        
        for hw_type, (min_s, max_s) in self.truck_speed_ranges.items():
            mask = (hw_col == hw_type) & (speed_col >= min_s) & (speed_col <= max_s)
            speed_scores[mask] = 1.0
        
        priority_scores = np.array([
            self.highway_priority.get(hw, 0) / 100.0 if pd.notna(hw) else 0.0
            for hw in hw_col
        ])
        
        distance_scores = np.where(
            gdf_labeled['dist_to_highway_m'].notna(),
            1 - (gdf_labeled['dist_to_highway_m'] / max_distance),
            0.0
        )

        total_weighted_score = (
            0.3 * speed_scores + 
            0.3 * priority_scores + 
            0.4 * distance_scores
        )
        
        return pd.Series(total_weighted_score, index=gdf_labeled.index)

    @staticmethod
    def _add_empty_highway_column(df):
        """Returns a copy of ``df`` with a ``highway_label`` column initialised to ``None``.

        Used as a safe fallback whenever a labeling step fails or produces no results.

        Args:
            df: Input trip DataFrame.

        Returns:
            A copy of ``df`` with an additional ``highway_label`` column set to ``None``.
        """
        df = df.copy()
        df["highway_label"] = None
        return df

    def _load_offline_map_with_pyogrio(self, country: str):
        """Loads the highway layer for a single country from its offline OSM file.

        On the first call the layer is read from the GeoPackage / PBF via
        *pyogrio* and cached as a Parquet file. Subsequent calls return the
        cached Parquet directly, skipping the expensive file read.

        Args:
            country: Country code (e.g. ``"DE"``). Used to look up the OSM
                filename in ``country_osm_map`` and to name the cache file.

        Returns:
            A :class:`~geopandas.GeoDataFrame` containing highway geometries and
            a ``highway`` attribute column, filtered to the types listed in
            ``highway_labels``.

        Raises:
            Exception: Re-raises any exception thrown by *pyogrio* after logging
            the error.
        """
        highway_where = "highway IN ('" + "','".join(self.highway_labels) + "')"
        output_path = self.save_loaded_map_path / f"{country}_highways.parquet"

        if output_path.exists():
            logger.info(f"Highways for {country} from offline map {self.country_osm_map[country]} were already loaded")
            return gpd.read_parquet(output_path)
        
        else:
            try:
                logger.info(f"Loading highways for {country} from offline map {self.country_osm_map[country]} as GeoDataFrame")
                gdf_highways = gpd.read_file(
                    str(self.map_path / self.country_osm_map[country]),
                    engine="pyogrio",
                    use_arrow=True,
                    layer="lines",
                    where=highway_where
                )

                highway_counts = gdf_highways['highway'].value_counts()
                logger.info(f"Highway type distribution:\n{highway_counts}")
                
                gdf_highways.to_parquet(output_path, engine='pyarrow')

                return gdf_highways

            except Exception as e:
                logger.error(f"Failed to load offline map {self.country_osm_map[country]} as GeoDataFrame: {e}")
                raise

    def _load_multiple_offline_maps_with_pyogrio(self, country: str):
        """Loads and concatenates highway layers for a comma-separated list of countries.

        Delegates to :meth:`_load_offline_map_with_pyogrio` for each individual
        country, then concatenates the results into a single GeoDataFrame.

        Args:
            country: Comma-separated country codes (e.g. ``"DK,SE"``).

        Returns:
            A concatenated :class:`~geopandas.GeoDataFrame` covering all
            specified countries, with a reset integer index.
        """
        gdfs = []
        for country in country.split(","):
            gdf = self._load_offline_map_with_pyogrio(country.strip())
            gdfs.append(gdf)
        
        gdf_concat = pd.concat(gdfs, ignore_index=True)
        
        
        return gdf_concat
      
    def _online_labeling_with_osmnx_features(
        self, 
        df: pd.DataFrame, 
        bbox_buffer: float, 
        max_distance: float, 
        country: str
    ) -> pd.DataFrame:
        """Labels GPS points using highway features fetched live from OSM via *osmnx*.

        Constructs a bounding box from the trip's coordinate extent (plus
        ``bbox_buffer``), downloads matching highway features with
        :func:`osmnx.features.features_from_bbox`, then performs a nearest-
        neighbour spatial join within ``max_distance`` to assign a label to
        each point.

        Args:
            df: Trip DataFrame with ``latitude_cval_ippc`` and
                ``longitude_cval_ippc`` columns.
            bbox_buffer: Padding in decimal degrees added to each side of the
                bounding box before querying OSM.
            max_distance: Maximum snap distance in metres. Points further from
                any highway remain unlabeled (``NaN``).
            country: Country code used for CRS lookup.

        Returns:
            A copy of ``df`` with a ``highway_label`` column. If the OSM
            request or spatial join fails, all labels are ``None``.
        """
        lon_min = df["longitude_cval_ippc"].min() - bbox_buffer
        lon_max = df["longitude_cval_ippc"].max() + bbox_buffer
        lat_min = df["latitude_cval_ippc"].min() - bbox_buffer
        lat_max = df["latitude_cval_ippc"].max() + bbox_buffer
        bbox = (lon_min, lat_min, lon_max, lat_max)
        
        try:
            gdf_highways = features_from_bbox(bbox, tags={"highway": self.highway_labels})
        except Exception as e:
            logger.error(f"OSM fetch failed: {e}")
            return self._add_empty_highway_columns(df)

        if gdf_highways.empty:
            logger.warning("No highways found in bbox")
            return self._add_empty_highway_columns(df)

        # Filter valid line geometries
        gdf_highways = gdf_highways[
            gdf_highways.geometry.notna() &
            gdf_highways.geometry.is_valid &
            gdf_highways.geometry.type.isin(['LineString', 'MultiLineString'])
        ].copy()

        if gdf_highways.empty:
            logger.warning("No valid highway geometries")
            return self._add_empty_highway_columns(df)

        # Create point GeoDataFrame
        df = df.copy()
        df['_point_id'] = range(len(df))
        gdf_points = gpd.GeoDataFrame(
            df,
            geometry=gpd.points_from_xy(df.longitude_cval_ippc, df.latitude_cval_ippc),
            crs="EPSG:4326"
        )

        # Project to UTM for accurate distances
        try:
            gdf_points = gdf_points.to_crs(self.utm_codes[country])
            gdf_highways = gdf_highways.to_crs(self.utm_codes[country])
        except Exception as e:
            logger.error(f"CRS projection failed: {e}")
            return self._add_empty_highway_columns(df)

        # Spatial join with distance threshold
        try:
            gdf_labeled = gpd.sjoin_nearest(
                gdf_points,
                gdf_highways[['geometry', 'highway']],
                how='left',
                max_distance=max_distance,
                distance_col='dist_to_highway_m'
            )
        
            gdf_labeled = gdf_labeled.sort_values('dist_to_highway_m').drop_duplicates(
                subset=['_point_id'], keep='first'
            )
        except Exception as e:
            logger.error(f"Spatial join failed: {e}")
            return self._add_empty_highway_columns(df)

        result = df.merge(
            gdf_labeled[['_point_id', 'highway']],
            on='_point_id',
            how='left'
        )
        result = result.drop(columns=['_point_id', 'dist_to_highway_m'])
        result = result.rename(columns={'highway': 'highway_label'})

        assert len(result) == len(df), "Row count mismatch after labeling"
        
        return result

    def _online_labeling_with_osmnx_graph(
        self, 
        df: pd.DataFrame, 
        bbox_buffer: float = 0.01
    ) -> pd.DataFrame:
        """Labels GPS points by snapping them to the nearest edge of an OSM street graph.

        Downloads a routable graph for the trip's bounding box using
        :func:`osmnx.graph.graph_from_bbox`, then snaps every GPS point to its
        nearest graph edge with :func:`osmnx.distance.nearest_edges` to extract
        the highway type.

        Args:
            df: Trip DataFrame with ``latitude_cval_ippc`` and
                ``longitude_cval_ippc`` columns.
            bbox_buffer: Padding in decimal degrees added to each side of the
                bounding box before downloading the graph. Defaults to ``0.01``.

        Returns:
            A copy of ``df`` with a ``highway_label`` column. If graph
            download fails or no edges are found, all labels are ``None``.
        """
        lon_min = df["longitude_cval_ippc"].min() - bbox_buffer
        lon_max = df["longitude_cval_ippc"].max() + bbox_buffer
        lat_min = df["latitude_cval_ippc"].min() - bbox_buffer
        lat_max = df["latitude_cval_ippc"].max() + bbox_buffer
        bbox = (lon_min, lat_min, lon_max, lat_max)

        # Custom filter for highways
        custom_filter = f'["highway"~"{"|".join(self.highway_labels)}"]'
        df_unlabeled_lenght = len(df)

        try:
            # Fetch the graph from OSM
            G = graph_from_bbox(
                bbox,
                simplify=True,
                retain_all=False,
                truncate_by_edge=False,
                custom_filter=custom_filter
            )

        except Exception as e:
            logger.error(f"OSM fetch failed: {e}")
            return self._add_empty_highway_columns(df)

        if len(G.edges) == 0:
            logger.warning("No highways found in graph")
            return self._add_empty_highway_columns(df)

        # Snap each point to the nearest edge
        lons = df['longitude_cval_ippc'].to_numpy()
        lats = df['latitude_cval_ippc'].to_numpy()

        # Find the nearest edges for each point
        edges = nearest_edges(G, X=lons, Y=lats)

        # Extract highway labels from the nearest edges
        df['highway_label'] = [
            h[0] if isinstance(h, list) else h
            for h in (G[u][v][k].get('highway', None) for u, v, k in edges)
        ]

        assert len(df) == df_unlabeled_lenght, f"Row mismatch: {len(df)} != {len(df_unlabeled_lenght)}"

        return df

    def _online_labeling_with_overpass(
        self, 
        df: pd.DataFrame, 
        bbox_buffer: float, 
        max_distance: float, 
        country: str
    ) -> pd.DataFrame:
        """Labels GPS points using highway geometries fetched from the Overpass API.

        Queries the Overpass API for all highway ways within the trip's bounding
        box, builds a :class:`~geopandas.GeoDataFrame` of road geometries, and
        then scores every (point, candidate highway) pair using a combination of
        priority and distance. Also detects roundabouts and intersections.

        Args:
            df: Trip DataFrame with ``latitude_cval_ippc``, ``longitude_cval_ippc``,
                and ``vehspd_cval_cpc`` columns.
            bbox_buffer: Padding in decimal degrees added to each side of the
                bounding box before querying Overpass.
            max_distance: Maximum snap distance in metres. Matches beyond this
                threshold are discarded.
            country: Country code used for CRS projection lookup.

        Returns:
            A copy of ``df`` with four additional columns:

            * ``highway_label`` – OSM highway type string or ``None``.
            * ``is_roundabout`` – ``True`` if the matched way is a roundabout.
            * ``is_intersection`` – ``True`` if ≥ 3 distinct OSM ways are within 20 m.
            * ``dist_to_highway_m`` – Distance in metres to the matched highway.

            Falls back to a single ``highway_label = None`` column on any error.
        """   
        lon_min = df["longitude_cval_ippc"].min() - bbox_buffer
        lon_max = df["longitude_cval_ippc"].max() + bbox_buffer
        lat_min = df["latitude_cval_ippc"].min() - bbox_buffer
        lat_max = df["latitude_cval_ippc"].max() + bbox_buffer
        
        way_queries = []
        for hwy_type in self.highway_labels:
            way_queries.append(f'  way["highway"="{hwy_type}"]({lat_min},{lon_min},{lat_max},{lon_max});')
        
        overpass_query = f"""
            [out:json][timeout:60];
            (
            {chr(10).join(way_queries)}
            );
            out geom;
            """
        
        # Fetch highways
        try:
            response = requests.post(
                "https://overpass-api.de/api/interpreter",
                data=overpass_query,
                timeout=90
            )
            response.raise_for_status()
            data = response.json()
            
            features = []
            for element in data.get('elements', []):
                if element.get('type') == 'way' and 'geometry' in element:
                    coords = [(node['lon'], node['lat']) for node in element['geometry']]
                    if len(coords) >= 2:
                        tags = element.get('tags', {})
                        hwy_type = tags.get('highway', 'unknown')
                        
                        is_roundabout = (
                            tags.get('junction') == 'roundabout' or
                            tags.get('junction') == 'circular'   or
                            tags.get('junction') == 'yes'
                        )
                        
                        features.append({
                            'geometry': LineString(coords),
                            'highway': hwy_type,
                            'priority': self.highway_prio.get(hwy_type, 0),
                            'is_roundabout': is_roundabout,
                            'osm_id': element.get('id')
                        })
            
            if not features:
                logger.warning("No highways found in bbox")
                return self._add_empty_highway_column(df)
            
            gdf_highways = gpd.GeoDataFrame(features, crs="EPSG:4326")
            logger.info(f"✓ Fetched {len(gdf_highways)} highways")
            
        except Exception as e:
            logger.error(f"Overpass fetch failed: {e}")
            return self._add_empty_highway_column(df)

        if gdf_highways.empty:
            return self._add_empty_highway_column(df)

        # Filter valid geometries
        gdf_highways = gdf_highways[
            gdf_highways.geometry.notna() &
            gdf_highways.geometry.is_valid &
            gdf_highways.geometry.type.isin(['LineString', 'MultiLineString'])
        ].copy()

        if gdf_highways.empty:
            return self._add_empty_highway_column(df)

        gdf_points = gpd.GeoDataFrame(
            df,
            geometry=gpd.points_from_xy(df.longitude_cval_ippc, df.latitude_cval_ippc),
            crs="EPSG:4326"
        )

        # Project to EPSG:3035
        try:
            gdf_points = gdf_points.to_crs(self.crs_codes[country])
            gdf_highways = gdf_highways.to_crs("EPSG:3035")
        except Exception as e:
            logger.error(f"CRS projection failed: {e}")
            return self._add_empty_highway_column(df)

        # Find ALL nearby highways for each point (not just one!)
        try:
            # Buffer points to find ALL candidates within max_distance
            gdf_points_buffered = gdf_points.copy()
            gdf_points_buffered['geometry'] = gdf_points_buffered.geometry.buffer(max_distance)
            
            # Get ALL highways that intersect with buffered points
            gdf_all_candidates = gpd.sjoin(
                gdf_points_buffered,
                gdf_highways[['geometry', 'highway', 'priority', 'is_roundabout', 'osm_id']],
                how='left',
                predicate='intersects'
            )
            
            # Calculate actual distance for each candidate
            distances = []
            for idx, row in gdf_all_candidates.iterrows():
                if pd.notna(row['highway']):
                    # idx is the original point index
                    point_geom = gdf_points.loc[idx].geometry
                    # Find highway by osm_id
                    highway_match = gdf_highways[gdf_highways['osm_id'] == row['osm_id']]
                    if len(highway_match) > 0:
                        dist = point_geom.distance(highway_match.geometry.iloc[0])
                        distances.append(dist)
                    else:
                        distances.append(None)
                else:
                    distances.append(None)
            
            gdf_all_candidates['dist_to_highway_m'] = distances
            
            # Filter by max_distance
            gdf_all_candidates = gdf_all_candidates[
                gdf_all_candidates['dist_to_highway_m'].isna() | 
                (gdf_all_candidates['dist_to_highway_m'] <= max_distance)
            ]
            
            logger.info(f"Found {len(gdf_all_candidates)} total candidates for {len(gdf_points)} points")
            
            # PRIORITY SCORING for all candidates
            mask_has_match = gdf_all_candidates['highway'].notna()
            
            if mask_has_match.any():
                # Calculate distance penalty
                gdf_all_candidates.loc[mask_has_match, 'distance_penalty'] = np.where(
                    gdf_all_candidates.loc[mask_has_match, 'dist_to_highway_m'] < 15,
                    2.0,
                    np.where(
                        gdf_all_candidates.loc[mask_has_match, 'dist_to_highway_m'] < 30,
                        1.0,
                        np.where(
                            gdf_all_candidates.loc[mask_has_match, 'dist_to_highway_m'] < 50,
                            0.5,
                            0.2
                        )
                    )
                )
                
                # Calculate score: priority * distance_penalty
                gdf_all_candidates.loc[mask_has_match, 'score'] = (
                    gdf_all_candidates.loc[mask_has_match, 'priority'] * 
                    gdf_all_candidates.loc[mask_has_match, 'distance_penalty']
                )
                
                # Pick HIGHEST score for each point (BEST match!)
                idx_best = gdf_all_candidates[mask_has_match].groupby(level=0)['score'].idxmax()
                gdf_labeled = gdf_all_candidates.loc[idx_best].copy()
                
                # Remove points that are too far from major roads
                low_score = (
                    (gdf_labeled['dist_to_highway_m'] > 50) & 
                    (gdf_labeled['priority'] < 85)
                ) | (
                    (gdf_labeled['dist_to_highway_m'] > 30) & 
                    (gdf_labeled['priority'] < 50)
                )
                
                gdf_labeled.loc[low_score, 'highway'] = None
                gdf_labeled.loc[low_score, 'is_roundabout'] = False
                gdf_labeled.loc[low_score, 'dist_to_highway_m'] = None
                
            else:
                # No matches at all
                gdf_labeled = gdf_points.copy()
                gdf_labeled['highway'] = None
                gdf_labeled['is_roundabout'] = False
                gdf_labeled['dist_to_highway_m'] = None
            
            # Detect intersections (count unique highways within 20m)
            gdf_labeled['is_intersection'] = False
            
            for idx in gdf_labeled.index:
                if pd.notna(gdf_labeled.loc[idx, 'highway']):
                    point_geom = gdf_points.loc[idx].geometry
                    nearby = gdf_highways[gdf_highways.distance(point_geom) <= 20]
                    if len(nearby['osm_id'].unique()) >= 3:
                        gdf_labeled.loc[idx, 'is_intersection'] = True
            
        except Exception as e:
            logger.error(f"Matching failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return self._add_empty_highway_columns(df)

        # Extract only the 3 new columns, preserve original index order
        result = df.copy()
        result['highway_label'] = gdf_labeled['highway']
        result['is_roundabout'] = gdf_labeled['is_roundabout'].fillna(False).astype(bool)
        result['is_intersection'] = gdf_labeled['is_intersection'].fillna(False).astype(bool)
        result['dist_to_highway_m'] = gdf_labeled['dist_to_highway_m']

        # Verify row count
        assert len(result) == len(df), f"Row mismatch! Expected {len(df)}, got {len(result)}"
        
        # Log statistics
        labeled_count = result['highway_label'].notna().sum()
        roundabout_count = result['is_roundabout'].sum()
        intersection_count = result['is_intersection'].sum()
        
        logger.info(f"✓ {labeled_count}/{len(result)} labeled ({labeled_count/len(result)*100:.1f}%)")
        logger.info(f"  Roundabouts: {roundabout_count} | Intersections: {intersection_count}")
        
        if labeled_count > 0:
            logger.info(f"\n{result['highway_label'].value_counts()}")
        
        return result
    
    def _offline_labeling_with_pyogrio(
        self, df: pd.DataFrame, 
        bbox_buffer: float, 
        max_distance: float,
        crs: str, 
        gdf_highways: gpd.GeoDataFrame
    ) -> pd.DataFrame:
        """Labels GPS points against a pre-loaded offline highway GeoDataFrame.

        Clips the highway GeoDataFrame to the trip's bounding box, projects
        both points and highways to the specified CRS, then runs a nearest-
        neighbour spatial join. Duplicate matches are resolved by the composite
        weighted score from :meth:`_calculate_weighted_score`.

        Args:
            df: Trip DataFrame with ``latitude_cval_ippc`` and
                ``longitude_cval_ippc`` columns.
            bbox_buffer: Padding in decimal degrees applied to the trip extent
                when clipping the highway GeoDataFrame.
            max_distance: Maximum snap distance in metres passed to
                :func:`geopandas.sjoin_nearest`.
            crs: EPSG string for the projected CRS used during distance
                calculations (e.g. ``"EPSG:3035"``).
            gdf_highways: Pre-loaded highway GeoDataFrame in ``EPSG:4326``.

        Returns:
            A copy of ``df`` with ``highway_label`` and ``dist_to_highway_m``
            columns. Falls back to ``highway_label = None`` on any error.
        """
        lon_min = df["longitude_cval_ippc"].min() - bbox_buffer
        lon_max = df["longitude_cval_ippc"].max() + bbox_buffer
        lat_min = df["latitude_cval_ippc"].min() - bbox_buffer
        lat_max = df["latitude_cval_ippc"].max() + bbox_buffer

        try:
            gdf_highways_bbox = gdf_highways.cx[lon_min:lon_max, lat_min:lat_max]
        except Exception as e:
            logger.error(f"Filtering failed: {e}")
            return self._add_empty_highway_column(df)
        
        try:
            gdf_highways_bbox = gdf_highways_bbox[
                gdf_highways_bbox.geometry.notna() &
                gdf_highways_bbox.geometry.is_valid
            ].copy()
            
            if len(gdf_highways_bbox) == 0:
                logger.warning("No valid highways in bbox")
                return self._add_empty_highway_column(df)
            else:
                logger.info(f"Valid geometries: {len(gdf_highways_bbox)} features")    
        except Exception as e:
            logger.error(f"Geometry validation failed: {e}")
            return self._add_empty_highway_column(df)

        df = df.copy()
        df['point_id'] = range(len(df))

        gdf_points = gpd.GeoDataFrame(
            df,
            geometry=gpd.points_from_xy(df.longitude_cval_ippc, df.latitude_cval_ippc),
            crs="EPSG:4326"
        )

        try:
            gdf_points = gdf_points.to_crs(crs)
            gdf_highways_bbox = gdf_highways_bbox.to_crs(crs)
        except Exception as e:
            logger.error(f"CRS projection failed: {e}")
            return self._add_empty_highway_column(df)

        try:
            gdf_labeled = gpd.sjoin_nearest(
                gdf_points,
                gdf_highways_bbox[["geometry", "highway"]],  
                how="left",
                max_distance=max_distance,
                distance_col="dist_to_highway_m",
            )

            # Calculate scores
            gdf_labeled['total_weighted_score'] = self._calculate_weighted_score(gdf_labeled, max_distance)

            # Sort & deduplicate
            gdf_labeled = (
                gdf_labeled.sort_values(['point_id', 'total_weighted_score'], ascending=[True, False])
                .drop_duplicates(subset=['point_id'], keep='first')
            )
            
        except Exception as e:
            logger.error(f"Spatial join failed: {e}")
            return self._add_empty_highway_column(df)

        result = df.merge(
            gdf_labeled[['point_id', 'highway', 'dist_to_highway_m']],
            on='point_id',
            how='left'
        )

        result = result.drop(columns=['point_id'])
        result = result.rename(columns={'highway': 'highway_label'})

        assert len(result) == len(df), f"Row mismatch: {len(result)} != {len(df)}"

        return result

    def _offline_labeling_with_pyrosm(
        self, 
        df: pd.DataFrame, 
        bbox_buffer: float, 
        max_distance: float, 
        country: str
    ) -> pd.DataFrame:
        """Labels GPS points using *pyrosm* to parse an offline OSM PBF file.

        Reads the bounding-box-clipped highway network from the OSM PBF for the
        given country, then performs a priority-weighted nearest-neighbour join
        to assign highway labels.

        Args:
            df: Trip DataFrame with ``latitude_cval_ippc`` and
                ``longitude_cval_ippc`` columns.
            bbox_buffer: Padding in decimal degrees added to the trip's bounding
                box before passing it to *pyrosm*.
            max_distance: Maximum snap distance in metres passed to
                :func:`geopandas.sjoin_nearest`.
            country: Country code used to look up the OSM PBF filename and the
                projected CRS.

        Returns:
            A copy of ``df`` with a ``highway_label`` column and the temporary
            ``dist_to_highway_m`` column dropped. Falls back to
            ``highway_label = None`` on any error.
        """
        lon_min = df["longitude_cval_ippc"].min() - bbox_buffer
        lon_max = df["longitude_cval_ippc"].max() + bbox_buffer
        lat_min = df["latitude_cval_ippc"].min() - bbox_buffer
        lat_max = df["latitude_cval_ippc"].max() + bbox_buffer
        bbox = (lon_min, lat_min, lon_max, lat_max)

        osm = OSM(str(self.map_path / self.country_osm_map[country]), bounding_box=bbox)
    
        try:

            logger.info("Building GeoDataframe")

            gdf_highways = osm.get_data_by_custom_criteria(
                custom_filter={'highway': self.highway_labels },
                filter_type='keep',
                osm_keys_to_keep=['highway'],
                tags_as_columns=['highway'],
                keep_nodes=False,     
                keep_ways=True,
                keep_relations=False
            )

            logger.info("GeoDataframe sucesfully build")
            
            del osm
            gc.collect()

        except Exception as e:
            logger.error(f"OSM fetch failed: {e}")
            return self._add_empty_highway_columns(gdf_highways)
        
        try:
            gdf_highways = gdf_highways[
                gdf_highways.geometry.notna() &
                gdf_highways.geometry.is_valid &
                gdf_highways.geometry.type.isin(['LineString', 'MultiLineString'])
            ].copy()
        except Exception as e:
            logger.error(f"No data: {e}")
            return self._add_empty_highway_columns(gdf_highways)
     
        # Copy df and create a unique point ID for merging after join
        df = df.copy()
        df['_point_id'] = range(len(df))

        # Build GeoDataFrame of points
        gdf_points = gpd.GeoDataFrame(
            df,
            geometry=gpd.points_from_xy(df.longitude_cval_ippc, df.latitude_cval_ippc),
            crs="EPSG:4326"  # original lat/lon CRS
        )

        try:
            # Reproject points and highways to projected CRS (meters) for distance calculations
            gdf_points = gdf_points.to_crs(self.utm_codes[country])
            gdf_highways = gdf_highways.to_crs(self.utm_codes[country])
        except Exception as e:
            logger.error(f"CRS projection failed: {e}")
            return self._add_empty_highway_columns(df)

        try:

            gdf_labeled = gpd.sjoin_nearest(
                gdf_points,
                gdf_highways[['geometry', 'highway']],  
                how='left',
                max_distance=max_distance,
                distance_col='dist_to_highway_m'
            )

            # Assign highway priority and keep the closest/highest priority highway per point
            gdf_labeled['highway_priority'] = gdf_labeled['highway'].map(self.highway_prio).fillna(0)
            gdf_labeled = gdf_labeled.sort_values(
                ['_point_id', 'highway_priority', 'dist_to_highway_m'],
                ascending=[True, False, True]
            ).drop_duplicates(subset=['_point_id'], keep='first')

        except Exception as e:
            logger.error(f"Spatial join failed: {e}")
            return self._add_empty_highway_columns(df)

        # Merge labeled highways back to original df
        result = df.merge(
            gdf_labeled[['_point_id', 'highway', 'dist_to_highway_m']],
            on='_point_id',
            how='left'
        )
        result = result.drop(columns=['_point_id', 'dist_to_highway_m'])
        result = result.rename(columns={'highway': 'highway_label'})

        # Ensure no rows are lost
        assert len(result) == len(df), "Row count mismatch after labeling"

        return result
    
    def _online_labeling_pipeline(
        self,
        bbox_buffer: float,
        max_distance: float,
        country: str,
        chunk_size: int,
        function: callable,
        datasets: Tuple[List[Path], List[str]]
    ) -> None:
        """Runs an online labeling function over a list of trip files.

        Iterates over the provided file paths, skips already-processed outputs,
        and calls ``function`` to label each trip. Trips larger than
        ``chunk_size`` rows are split into chunks that are labeled individually
        and then concatenated. A verification map is generated for every trip.

        Args:
            bbox_buffer: Padding in decimal degrees added to each trip's bounding
                box before the online query.
            max_distance: Maximum snap distance in metres.
            country: Country code forwarded to ``function`` and to the plot
                directory hierarchy.
            chunk_size: Maximum number of rows to process in a single call to
                ``function``. Larger files are split into chunks of this size.
            function: A callable with signature
                ``(df, bbox_buffer, max_distance, country) -> pd.DataFrame``
                that performs the actual labeling (e.g.
                :meth:`_online_labeling_with_overpass`).
            datasets: Tuple ``(paths, filenames)`` as returned by
                :meth:`_get_datasets` or :meth:`_get_datasets_by_country`.

        Returns:
            None
        """
        paths, filenames = datasets

        with tqdm(total=len(filenames), desc="Online highway labeling") as pbar:
            
            for path, filename in zip(paths, filenames):
                
                output_path = Path(self.save_labels_path, f"{filename}{self.file_extension}")
                if output_path.exists():
                    tqdm.write(f"Skipping {filename}: already processed")
                    pbar.update(1)
                    continue

                df = pd.read_parquet(path, columns=self.highway_labeling_signals)
                total_rows = len(df)

                if total_rows <= chunk_size:
                    df_labeled = function(df, bbox_buffer, max_distance, country)
                else:
                    n_chunks = (total_rows + chunk_size - 1) // chunk_size
                    chunks = np.array_split(df, n_chunks)
                    results = []
                    for chunk in tqdm(chunks, desc=f"Labeling highways for {filename}", unit="chunk"):
                        labeled_chunk = function(chunk, bbox_buffer, max_distance, country)
                        results.append(labeled_chunk)
                    df_labeled = pd.concat(results, ignore_index=True)

            self._plot_and_verify_labels(df_labeled, filename, country)

        return None
    
    def _offline_labeling_pipeline(
        self,
        bbox_buffer: float,
        max_distance: float,
        function: callable,
    ) -> None:
        """Runs an offline labeling function over all countries and their trip files.

        Groups trips by country using :meth:`_get_datasets_by_country`, loads
        the corresponding offline highway map once per country (or country
        combination), then applies ``function`` to each individual trip. Results
        are written to ``save_labels_path`` and a verification map is generated.
        The highway GeoDataFrame is released from memory after each country to
        limit peak RAM usage.

        Args:
            bbox_buffer: Padding in decimal degrees added to the trip's bounding
                box when clipping the highway GeoDataFrame.
            max_distance: Maximum snap distance in metres passed to ``function``.
            function: A callable with signature
                ``(df, bbox_buffer, max_distance, crs, gdf_highways) -> pd.DataFrame``
                that performs the actual labeling (e.g.
                :meth:`_offline_labeling_with_pyogrio`).

        Returns:
            None
        """
        datasets = self._get_datasets_by_country()
        
        with tqdm(total=sum(len(filenames) for _, (_, filenames) in datasets.items()), desc="Preprocessing datasets") as pbar:
            for country, (paths, filenames) in datasets.items():
                country = "DK,SE"
                if "," in country:
                    gdf_highways = self._load_multiple_offline_maps_with_pyogrio(country)
                    crs = self.crs_codes["EU"]
                else:
                    gdf_highways = self._load_offline_map_with_pyogrio(country)
                    crs = self.crs_codes[country]

                for path, filename in zip(paths, filenames):
                    output_path = Path(self.save_labels_path, f"{filename}{self.file_extension}")
                    if output_path.exists():
                        tqdm.write(f"Skipping {filename}: already processed")
                        pbar.update(1)
                        continue

                    df = pd.read_parquet(path, columns=self.highway_labeling_signals)
                    df_labeled = function(df, bbox_buffer, max_distance, crs, gdf_highways)
                    df_labeled.to_parquet(output_path)
                    self._plot_and_verify_labels(df_labeled, filename, country)
                    pbar.update(1)
                
                del gdf_highways
                gc.collect()
                
    def run(self):
        """Entry point for the highway labeling pipeline.

        Executes the offline labeling pipeline using
        :meth:`_offline_labeling_with_pyogrio` as the labeling backend, logs
        the total elapsed time, and returns.

        Returns:
            None
        """
        start_time = time.time()
        
        self._offline_labeling_pipeline(
            bbox_buffer=0.01,
            max_distance=200,
            function=self._offline_labeling_with_pyogrio,
        )
        
        elapsed_total = time.time() - start_time
        minutes, seconds = divmod(elapsed_total, 60)
        logger.info(f"Completed highway labeling in {int(minutes)} min {int(seconds)} sec")
    
