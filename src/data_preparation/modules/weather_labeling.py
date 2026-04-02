"""
Module for truck trip data processing.
"""

import logging
import time
from pathlib import Path
from typing import List, Tuple
import json

import numpy as np
import pandas as pd
from meteostat import Hourly, Point
from omegaconf import DictConfig
from tqdm import tqdm

from src.data_preparation.modules import constants as const

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

 
class TruckDataWeatherLabeler: 
    def __init__(self, cfg: DictConfig):
        self.data_path = Path(cfg.data.paths.validated_truck_data)
        self.file_extension = cfg.data.paths.file_extension
        
        self.save_path = Path(cfg.data.paths.weather_labels)
        self.save_path.mkdir(parents=True, exist_ok=True)
        
        self.results_path =  Path("llm-erange/results/weather_checks")
        self.results_path.mkdir(parents=True, exist_ok=True)
                                                                        
        self.signals = const.WEATHER_LABELING_SIGNALS
        self.window_size_km = cfg.data.weather_labeling_configs.window_size_km
        self.window_size_seconds = cfg.data.weather_labeling_configs.window_size_seconds

        self.weather_codes = const.METEOSTAT_TO_WEATHER
        self.surface_codes  =const.METEOSTAT_TO_SURFACE

    def _get_datasets(self) -> Tuple[List[Path], List[str]]:
        paths = sorted(self.data_path.glob(f"*{self.file_extension}"))
        filenames = [f.stem for f in paths]
        return paths, filenames
    
    def _create_windows(self, df: pd.DataFrame, use_distance: bool = False, use_time: bool = True) -> pd.DataFrame:
        df["is_first_row"] = False
        df["is_middle_row"] = False
        df["is_last_row"] = False

        if use_distance == use_time:
            raise ValueError("Exactly one of use_distance or use_time must be True")

        if use_distance:
            df["window_id"] = (df["hirestotalvehdist_cval_icuc"] // self.window_size_km).astype(np.int32)
        else:
            df["window_id"] = (np.arange(len(df)) // self.window_size_seconds).astype(np.int32)

        # Mark first, middle, last rows per window
        first_indices = df.groupby("window_id").apply(lambda g: g.index[0]).values
        middle_indices = df.groupby("window_id").apply(lambda g: g.index[len(g)//2]).values
        last_indices = df.groupby("window_id").apply(lambda g: g.index[-1]).values

        df.loc[first_indices, "is_first_row"] = True
        df.loc[middle_indices, "is_middle_row"] = True
        df.loc[last_indices, "is_last_row"] = True

        return df
    
    @staticmethod # dwpt fallback (10.1175/BAMS-86-2-225, 10.1175/1520-0450(1996)035<0601:IMFAOS>2.0.CO;2)
    def _dew_point(temp, rhum):
        a, b = 17.27, 237.7
        alpha = (a * temp) / (b + temp) + np.log(rhum / 100.0)
        return (b * alpha) / (a - alpha)
    
    def _fetch_hourly_weather_data(self, df: pd.DataFrame, time_delta_hour: int = 5) -> pd.DataFrame:
        """  
        Parameters:
            latitude (float): Latitude of the location in decimal degrees.
            longitude (float): Longitude of the location in decimal degrees.
            altitude (float): Altitude of the location in meters.
            timestamp (datetime-like): The datetime for which to fetch weather data.

        Returns:
            pandas.DataFrame: A DataFrame containing the hourly weather data for the specified
                            location and time range. columns include:

                            - station: only if query refers to multiple stations
                            - temp (°C): Air temperature
                            - dwpt (°C): Dew point
                            - rhum (%): Relative humidity
                            - prcp (mm): One-hour precipitation total
                            - snow (mm): Snow depth
                            - wdir (°): Average wind direction
                            - wspd (km/h): Average wind speed
                            - wpgt (km/h): Peak wind gust
                            - pres (hPa): Sea-level air pressure
                            - tsun (minutes): One-hour sunshine total
                            - coco (int/float): Weather condition code
        """
        results = []
        df["signal_time_hour"] = pd.to_datetime(df['signal_time']).dt.round('h')
        df["start_time_hour"] = df["signal_time_hour"] - pd.Timedelta(hours=time_delta_hour)
        df["end_time_hour"]   = df["signal_time_hour"] + pd.Timedelta(hours=time_delta_hour)

        for window_id, group in df.groupby("window_id"):
            success = False
            last_row = None  # Store the last valid row for fallback
            for role in ["is_middle_row", "is_first_row", "is_last_row"]:
                row = group[group[role]].squeeze()
                if isinstance(row, pd.Series) and not row.empty:
                    last_row = row  # Update last_row
                    try:
                        gps_point = Point(row.latitude_cval_ippc, row.longitude_cval_ippc, row.altitude_cval_ippc)
                        weather_interval = Hourly(gps_point, row.start_time_hour, row.end_time_hour).normalize().interpolate().fetch()
                    except Exception as e:
                        print(f"Weather fetch failed for window {window_id} at {row.signal_time_hour}: {e}")
                        weather_interval = None

                    if weather_interval is not None and not weather_interval.empty:
                        weather = weather_interval.loc[weather_interval.index == row.signal_time_hour]
                        weather['window_id'] = window_id
                        weather["airtempoutsd_cval_cpc"] = row.airtempoutsd_cval_cpc
                        results.append(weather)
                        success = True
                        break

            if not success and last_row is not None:
                weather = pd.DataFrame([{
                    "temp": np.nan, "dwpt": np.nan, "rhum": np.nan, "prcp": np.nan,
                    "snow": np.nan, "wdir": np.nan, "wspd": np.nan, "wpgt": np.nan,
                    "pres": np.nan, "tsun": np.nan, "coco": np.nan,
                    "window_id": window_id, "airtempoutsd_cval_cpc": last_row.airtempoutsd_cval_cpc
                }])

                results.append(weather)
        
        weather_df = pd.concat(results, ignore_index=True)
        
        weather_df["temp"] = weather_df["temp"].fillna(weather_df["airtempoutsd_cval_cpc"])

        mask = weather_df["dwpt"].isna() & weather_df["temp"].notna() & weather_df["rhum"].notna()
        if mask.any():
            weather_df.loc[mask, "dwpt"] = self._dew_point(
                weather_df.loc[mask, "temp"],
                weather_df.loc[mask, "rhum"]
            )

        return weather_df
      
    def _label_weather_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Label weather data with scientifically-grounded thresholds.
        Fills small gaps ONLY when surrounding values match (safe interpolation).
        
        References:
        - Meteostat codes: https://dev.meteostat.net/formats.html
        - WMO Manual on Codes (WMO-No. 306, Volume I.2)
        - Freezing rain: WMO (2017) via UNDRR
        - Black ice: American Meteorological Society Glossary
        - Road icing: Liu et al. (2023), Frontiers in Earth Science
        - Precipitation intensity: WMO present weather codes
        - Wind classifications: Beaufort scale adapted for meteorological use
        """
        temp = df["airtempoutsd_cval_cpc"]
        dwpt = df["dwpt"]
        rhum = df["rhum"]
        prcp = df["prcp"]
        snow = df["snow"]
        wspd = df["wspd"]
        wpgt = df["wpgt"]
        pres = df["pres"]
        tsun = df["tsun"]
        coco = df["coco"]

        # ============================================================================
        # AIR TEMPERATURE CLASSIFICATION
        # ============================================================================
        air_temperature = pd.Series(index=df.index, dtype="string")
        air_temperature[temp < -10] = "extreme_cold"
        air_temperature[(temp >= -10) & (temp <= 0)] = "freezing_cold"
        air_temperature[(temp > 0) & (temp <= 10)] = "cold"
        air_temperature[(temp > 10) & (temp <= 20)] = "moderate"
        air_temperature[(temp > 20) & (temp <= 30)] = "warm"
        air_temperature[temp > 30] = "extreme_heat"

        # ===========================================================================
        # ATMOSPHERIC WEATHER CLASSIFICATION  
        # Priority: Meteostat code > severe > precip > wind > fog > clear
        # ============================================================================
        weather = pd.Series(index=df.index, dtype="string")
        
        weather = coco.map(self.weather_codes)
        weather_na = weather.isna()
        
        # Storm criteria: Very high winds + low pressure OR heavy precip
        # Reference: WMO storm classifications
        severe_wind = (wpgt >= 120) | (wspd >= 90)  # ≥ Beaufort 10
        storm_pressure = severe_wind & ((prcp >= 10) | (pres <= 995))
        weather[storm_pressure & weather_na] = "storm"
        weather_na = weather.isna()
        
        # Heavy snow
        # Reference: WMO - Heavy snow ≥ 4cm/hr (≈4mm water equivalent/hr)
        heavy_snow_cond = (snow >= 4) & (temp <= 2) & weather_na
        weather[heavy_snow_cond] = "heavy_snow"
        weather_na = weather.isna()
        
        # Moderate/light snow
        # Reference: WMO - Snow 0.1-4mm/hr water equivalent
        snow_cond = (snow >= 0.1) & (snow < 4) & (temp <= 2) & weather_na
        weather[snow_cond] = "snow"
        weather_na = weather.isna()
        
        # Heavy rain
        # Reference: WMO - Heavy rain ≥ 10mm/hr
        heavy_rain_cond = (prcp >= 10) & (temp > 2) & weather_na
        weather[heavy_rain_cond] = "heavy_rain"
        weather_na = weather.isna()
        
        # Moderate rain
        # Reference: WMO - Moderate rain 2.5-10mm/hr
        rain_cond = (prcp >= 2.5) & (prcp < 10) & (temp > 2) & weather_na
        weather[rain_cond] = "rain"
        weather_na = weather.isna()
        
        # Light rain/drizzle
        # Reference: WMO - Light rain 0.5-2.5mm/hr, drizzle < 0.5mm/hr
        light_rain_cond = (prcp >= 0.1) & (prcp < 2.5) & (temp > 2) & weather_na
        weather[light_rain_cond] = "rain"
        weather_na = weather.isna()
        
        # High wind (without precip)
        # Reference: Beaufort scale - Gale force 8+ (≥62 km/h)
        high_wind_cond = (
            ((wpgt >= 75) | (wspd >= 62)) & 
            (prcp < 2.5) & 
            (snow < 1) & 
            weather_na
        )
        weather[high_wind_cond] = "high_wind"
        weather_na = weather.isna()
        
        # Fog
        # Reference: WMO - Fog is visibility < 1km
        # Indicators: RH ≥ 90%, small temp-dewpoint spread, low sunshine, light wind
        fog_cond = (
            (rhum >= 90) & 
            (abs(temp - dwpt) <= 2.5) &    # Small spread
            (tsun < 15) &                   # < 15 min sunshine/hour
            (prcp < 0.5) &                  # Not raining
            (wspd < 15) &                   # Light wind (fog disperses in wind)
            weather_na
        )
        weather[fog_cond] = "fog"
        weather_na = weather.isna()
        
        # Clear/fair conditions (default fallback)
        clear_cond = (
            (prcp <= 0.1) & 
            (snow <= 0.1) & 
            (wpgt < 50) & 
            (wspd < 40) &
            weather_na
        )
        weather[clear_cond] = "clear"
        
        #
        # GAP FILLING
        weather = self._fill_matching_gaps(weather, max_gap_size=5)
        df["weather_gap_filled"] = weather.notna() & coco.map(self.weather_codes).isna()
        df["air_temperature"] = air_temperature
        df["weather"] = weather

        return df
    
    @staticmethod
    def _fill_matching_gaps(series: pd.Series, max_gap_size: int = 3) -> pd.Series:
       
        series = series.copy()
        
        # Identify gap groups
        is_gap = series.isna()
        gap_groups = (is_gap != is_gap.shift()).cumsum()
        
        # Process each gap
        for gap_id in gap_groups[is_gap].unique():
            gap_mask = (gap_groups == gap_id) & is_gap
            gap_indices = gap_mask[gap_mask].index
            
            # Skip if gap too large
            if len(gap_indices) > max_gap_size:
                continue
            
            # Get surrounding values
            first_gap_idx = gap_indices[0]
            last_gap_idx = gap_indices[-1]
            
            # Find previous non-NaN value
            prev_idx = first_gap_idx - 1
            if prev_idx < 0 or prev_idx not in series.index:
                continue
            prev_value = series.iloc[series.index.get_loc(prev_idx)]
            
            # Find next non-NaN value  
            next_idx = last_gap_idx + 1
            if next_idx >= len(series) or next_idx not in series.index:
                continue
            next_value = series.iloc[series.index.get_loc(next_idx)]
            
            # Fill ONLY if previous and next match
            if pd.notna(prev_value) and pd.notna(next_value) and prev_value == next_value:
                series.loc[gap_indices] = prev_value
        
        return series

    @staticmethod
    def _label_truck_data(truck_data: pd.DataFrame, weather_data_labeled: pd.DataFrame) -> pd.DataFrame:
        truck_data_labeled = truck_data.merge(
            weather_data_labeled[[
                "window_id", 
                "air_temperature",
                "weather"]], 
            on="window_id",
            how="left"
        )

        truck_data_labeled = truck_data_labeled.drop(columns=["window_id", 'is_middle_row'])

        return truck_data_labeled
    

    def _save_label_distribution(self, df_labeled: pd.DataFrame, filename: str) -> None:
        """
        Saves label distributions for air_temperature and weather to JSON.
        """
        distribution = {}
        for col in ["air_temperature", "weather"]:
            if col in df_labeled.columns:
                dist = df_labeled[col].value_counts(normalize=True) * 100
                distribution[col] = dist.to_dict()
        
        save_path = self.results_path / f"{filename}.json" 
        with open(save_path, "w") as f:
            json.dump(distribution, f, indent=4)

        return None

    def run(self):

        start_time = time.time()
        paths, filenames = self._get_datasets()
       
        with tqdm(total=len(filenames), desc="Preprocessing datasets") as pbar:
            for i, (path, filename) in enumerate(zip(paths, filenames)):
                output_path = Path(self.save_path, f"{filename}{self.file_extension}")
                if output_path.exists():
                    tqdm.write(f"Skipping {filename}: already processed")
                    pbar.update(1)
                    continue

                df = pd.read_parquet(path)
                df_windows = self._create_windows(df)
                weather_data = self._fetch_hourly_weather_data(df_windows)
                weather_data_labeled = self._label_weather_data(weather_data)
                df_labeled = self._label_truck_data(df, weather_data_labeled)
                df_labeled.to_parquet(output_path)
                self._save_label_distribution(df_labeled, filename)

                assert len(df) == len(df_labeled)

                pbar.update(1)
            
        elapsed_total = time.time() - start_time
        minutes, seconds = divmod(elapsed_total, 60)
        logger.info(f"Completed processing all {len(paths)} datasets in {int(minutes)} min {int(seconds)} sec")
          
            
