"""Electric Truck Data Quality Verification Module.
 
This module validates processed electric truck trip datasets before
they are used in the data preparation stage. It performs structural checks,
feature validation, temporal consistency verification, and trajectory visualization.
 
Example:
    >>> from omegaconf import OmegaConf
    >>> cfg = OmegaConf.load("configs/data_understanding.yaml")
    >>> verificator = TruckDataQualityVerificator(cfg)
    >>> verificator.run()
"""

import logging
from pathlib import Path
from typing import Tuple, List
import time

import pandas as pd
from omegaconf import DictConfig
from tqdm import tqdm
import folium
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

from src.data_understanding.modules import constants as const

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

class TruckDataQualityVerificator:
    """Performs quality verification for electric truck trip datasets.
 
    Loads processed trip files, validates schema consistency,
    checks temporal sampling constraints, and generates verification
    reports and trajectory visualizations.
 
    Attributes:
        cfg (DictConfig): Hydra configuration object (OmegaConf ``DictConfig``).
        file_extension (str): File suffix used when globbing trip files (e.g. ``'.parquet'``).
        vehicle_id_to_group (dict[str, int]): Mapping from vehicle identifier to vehicle group.
        vehicle_ids (list[str]): Supported vehicle identifiers.
        signal_mapping (dict[str, int]): Expected signals and their column ordering.
        processed_data_dir (Path): Directory containing processed datasets.
        verified_data_dir (Path): Directory where verified parquet files are written.
        results_dir (Path): Directory where verification reports are saved.
        plots_dir (Path): Directory where trajectory HTML plots are saved.
    """
    def __init__(self, cfg: DictConfig) -> None:
        """Initializes TruckDataQualityVerificator from a Hydra configuration object.
 
        Args:
            cfg (DictConfig): Hydra configuration object (OmegaConf ``DictConfig``).
                Must contain a ``data_understanding`` sub-config with the fields
                ``file_extension``, ``processed_data_dir``, ``verified_data_dir``,
                and ``results_dir``.
        """
        self.cfg = cfg
        self.file_extension = self.cfg.data_understanding.file_extension
        self.vehicle_id_to_group = const.VEHICLE_ID_TO_GROUP
        self.vehicle_ids = const.VEHICLE_IDS
        self.signal_mapping = const.SIGNAL_MAPPING
        self.processed_data_dir = Path(self.cfg.data_understanding.processed_data_dir)
        self.verified_data_dir = Path(self.cfg.data_understanding.verified_data_dir)
        self.results_dir =  Path(self.cfg.data_understanding.results_dir)
        self.plots_dir = self.results_dir/ "highway_plots"
        self.setup_dirs()

    def setup_dirs(self) -> None:
        """Creates required output directories if they do not already exist."""
        for path in [self.verified_data_dir, self.results_dir, self.plots_dir]:
            path.mkdir(parents=True, exist_ok=True)
    
    def get_datasets(self) -> Tuple[List[Path], List[str]]:
        """Gets all trip files in the verified data directory.

        Returns:
            Tuple[List[Path], List[str]]: A tuple of ``(paths, filenames)`` where 
            ``paths`` is a sorted list of :class:`pathlib.Path` objects 
            matching ``file_extension``, and ``filenames`` contains the 
            corresponding stems (no extension).
        """
        paths = sorted(self.processed_data_dir.glob(f"*{self.file_extension}"))
        filenames = [f.stem for f in paths]
        return paths, filenames

    def plot_trip_trajectory(self, df: pd.DataFrame, filename: str) -> None:
        """Renders an interactive Folium map with trip trajectory and key statistics.
 
        Draws the full GPS trajectory as a black polyline with start and end
        markers. A dashboard overlay displays vehicle ID, group, distance,
        average speed, SoC, and mean battery temperature. The map is saved
        to ``plots_dir``.
 
        Args:
            df (pd.DataFrame): Verified trip DataFrame. Must contain 
            ``latitude_cval_ippc``, ``longitude_cval_ippc``, ``v_id``, 
            ``v_group``, ``hirestotalvehdist_cval_icuc``, 
            ``vehspd_cval_cpc``, ``hv_bat_soc_cval_bms1``, and ``hv_batavcelltemp_cval_bms1`` columns.
            filename (str): Stem of the source file; used as the map title
                and output filename.
 
        Note:
            This function was developed with the assistance of Claude AI (Anthropic).
        """
        font_path_normal = "llm-erange/src/utils/times.ttf"
        font_path_bold   = "llm-erange/src/utils/times_bold.ttf"
        fm.fontManager.addfont(font_path_normal)
        fm.fontManager.addfont(font_path_bold)
        prop_normal = fm.FontProperties(fname=font_path_normal)
        plt.rcParams["font.family"] = prop_normal.get_name()
        plt.rcParams["font.size"] = 12
        
        m = folium.Map(
            location=[df["latitude_cval_ippc"].mean(), df["longitude_cval_ippc"].mean()],
            zoom_start=11
        )

        coords = list(zip(df["latitude_cval_ippc"], df["longitude_cval_ippc"]))
        folium.PolyLine(coords, color="#000000", weight=4, opacity=0.8).add_to(m)

        folium.Marker(
            (df.iloc[0]["latitude_cval_ippc"], df.iloc[0]["longitude_cval_ippc"]),
            tooltip="start",
            icon=folium.Icon(color="green", icon="play")
        ).add_to(m)
        folium.Marker(
            (df.iloc[-1]["latitude_cval_ippc"], df.iloc[-1]["longitude_cval_ippc"]),
            tooltip="end",
            icon=folium.Icon(color="red", icon="stop")
        ).add_to(m)

        # Stats
        vehicle_id = df["v_id"].iloc[0]
        vehicle_group = df["v_group"].iloc[0]
        distance_km = df["hirestotalvehdist_cval_icuc"].iloc[-1]
        avg_speed = df["vehspd_cval_cpc"].mean()
        start_soc = df["hv_bat_soc_cval_bms1"].iloc[0]
        end_soc = df["hv_bat_soc_cval_bms1"].iloc[-1]
        avg_cell_temp = df["hv_batavcelltemp_cval_bms1"].mean()

        # Dashboard
        legend_html = f'''
        <div style="position: fixed; 
                    top: 50px; 
                    right: 50px; 
                    background-color: white; 
                    border: 2px solid #333; 
                    z-index: 9999; 
                    font-family: 'Times New Roman';
                    font-size: 16px; 
                    padding: 15px; 
                    border-radius: 5px; 
                    box-shadow: 2px 2px 6px rgba(0,0,0,0.3);
                    width: auto;
                    min-width: 300px;">
            <p style="margin: 0 0 12px 0; 
                    border-bottom: 2px solid #333; 
                    padding-bottom: 8px;
                    font-family: 'Times New Roman';
                    font-size: 16px;">
                <strong>{filename}</strong> | 🟢 Start | 🔴 End
            </p>
            <p style="margin: 3px 0; font-size: 16px;">
                Vehicle ID: {vehicle_id}
            </p>
            <p style="margin: 3px 0; font-size: 16px;">
                Vehicle Group: {vehicle_group}
            </p>
            <hr style="margin: 10px 0; border: none; border-top: 1.3px solid #999;">
            <p style="margin: 3px 0; font-size: 16px;">
                Covered distance: {distance_km:.3f} km
            </p>
            <p style="margin: 3px 0; font-size: 16px;">
                Average speed: {avg_speed:.3f} km/h
            </p>
            <p style="margin: 3px 0; font-size: 16px;">
                Start SoC: {start_soc:.3f}%
            </p>
            <p style="margin: 3px 0; font-size: 16px;">
                End SoC: {end_soc:.3f}%
            </p>
            <p style="margin: 3px 0; font-size: 16px;">
                Mean battery cell temperature: {avg_cell_temp:.3f}°C
            </p>
        </div>
        '''
        
        m.get_root().html.add_child(folium.Element(legend_html))
        m.save(self.plots_dir / f"{filename}_stats.html")
       
    def add_vehicle_id_and_group(self, df: pd.DataFrame, filename: str) -> pd.DataFrame:
        """Extracts and assigns vehicle ID and group from the trip filename.
 
        Parses the filename to identify the vehicle identifier and looks up
        the corresponding vehicle group from ``vehicle_id_to_group``.
 
        Args:
            df (pd.DataFrame): Trip DataFrame to annotate. 
            filename (str): Stem of the source file containing the vehicle ID.
 
        Returns:
            pd.DataFrame: Input DataFrame with added ``v_id`` and ``v_group`` columns.
 
        Raises:
            ValueError: If no valid vehicle ID is found in the filename.
            ValueError: If multiple vehicle IDs are found in the filename.
        """
        filename_parts = filename.lower().split("_")
        v_id_matches = [v_id for v_id in self.vehicle_ids if v_id in filename_parts]

        if not v_id_matches:
            raise ValueError(f"No valid vehicle ID found in filename: {filename}")

        if len(v_id_matches) > 1:
            raise ValueError(
                f"Multiple vehicle IDs found in filename: {filename}, matches: {v_id_matches}"
            )

        v_id = v_id_matches[0]
        v_group = self.vehicle_id_to_group[v_id]
        
        df["v_id"] = v_id
        df["v_group"] = v_group
    
        return df
    
    def verify_expected_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validates that the DataFrame contains exactly the expected signal columns.
 
        Checks for missing and unexpected columns against ``signal_mapping``,
        then reorders columns to match the expected ordering.
 
        Args:
            df (pd.DataFrame): Trip DataFrame to validate.
 
        Returns:
            pd.DataFrame: Input DataFrame with columns reordered to match ``signal_mapping``.
 
        Raises:
            ValueError: If any expected signals are missing from the DataFrame.
            ValueError: If any unexpected signals are present in the DataFrame.
        """
        expected_signals = set(self.signal_mapping.keys())
        actual_signals = set(df.columns)
        
        missing_signals = expected_signals - actual_signals
        if missing_signals:
            raise ValueError(f"Missing signals in dataframe: {missing_signals}")
        
        extra_signals = actual_signals - expected_signals
        if extra_signals:
            raise ValueError(f"Unexpected signals in dataframe: {extra_signals}")
        
        ordered_columns = sorted(self.signal_mapping, key=lambda k: self.signal_mapping[k])
        df = df[ordered_columns]

        assert len(df.columns) == len(self.signal_mapping)

        return df
    
    def check_format_and_missing_data(self, df: pd.DataFrame, filename: str) -> pd.DataFrame:
        """Checks for missing values and validates 1 Hz temporal sampling.
 
        Verifies that all signal timestamps are spaced exactly 1 second apart
        and that no NaN values are present. Returns a single-row report
        DataFrame summarising the findings.
 
        Args:
            df (pd.DataFrame): Trip DataFrame to validate. Must contain a 
            ``signal_time`` column parseable as datetime.
            filename (str): Stem of the source file used as the report identifier.
 
        Returns:
            pd.DataFrame: Single-row report with columns ``filename``, ``valid``,
                ``has_missing_values``, ``is_1hz``, and ``wrong_index_groups``.
        """
        has_missing = df.isnull().values.any()

        time_diffs = pd.to_datetime(df["signal_time"]).diff().dt.total_seconds()

        wrong_indices = time_diffs[time_diffs != 1].index
        if not wrong_indices.empty:
            wrong_indices = sorted(set(i for idx in wrong_indices for i in (idx - 1, idx)))
        else:
            wrong_indices = []

        groups = []
        current_group = []
        for idx in wrong_indices:
            if not current_group or idx == current_group[-1] + 1:
                current_group.append(idx)
            else:
                groups.append(current_group)
                current_group = [idx]
        if current_group:
            groups.append(current_group)

        is_1hz = len(groups) == 0
        is_valid = not has_missing and is_1hz

        report = pd.DataFrame([{
            "filename": filename,
            "valid": is_valid,
            "has_missing_values": has_missing,
            "is_1hz": is_1hz,
            "wrong_index_groups": groups
        }])

        return report

    def run(self) -> None:
        """Runs the full quality verification pipeline.
 
        For each trip file, the pipeline:
 
        1. Assigns vehicle ID and group from the filename
        2. Validates expected signal columns and ordering
        3. Checks for missing values and 1 Hz sampling
        4. Saves the verified parquet file to ``verified_data_dir``
        5. Generates a trajectory HTML plot in ``plots_dir``
 
        After processing all files, failed verification reports are saved
        to ``results_dir/failed_quality_verification.json``.
        """
        start_time = time.time()
        paths, filenames = self._get_datasets()
        reports = []
 
        for path, filename in tqdm(zip(paths, filenames), total=len(paths),
                                   desc="Quality verification"):
            df = pd.read_parquet(path, engine="pyarrow")
            df = self._add_vehicle_id_and_group(df, filename)
            df = self._verify_expected_features(df)
            report = self._check_format_and_missing_data(df, filename)
            reports.append(report)
            df.to_parquet(self.verified_data_dir / f"{filename}.parquet", engine="pyarrow")
            self._plot_trip_trajectory(df, filename)
 
        # Combine all reports
        reports = pd.concat(reports, ignore_index=True)
 
        # Save failed verification reports
        failed_reports = reports[~reports["valid"]]
        failed_report_path = self.results_dir / "failed_quality_verification.json"
        failed_reports.to_json(
            failed_report_path,
            orient="records",
            lines=True,
        )
 
        elapsed_total = time.time() - start_time
        minutes, seconds = divmod(elapsed_total, 60)
        logger.info(
            f"Completed processing {len(paths)} datasets in {int(minutes)} min {int(seconds)} sec"
        )


            


