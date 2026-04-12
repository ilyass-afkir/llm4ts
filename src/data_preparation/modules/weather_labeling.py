"""Electric Truck Weather Labeling Module.

This module segments verified electric truck trip datasets into time or
distance windows, fetches hourly weather observations from Meteostat for
each window, classifies weather conditions and air temperatures using
meteorological thresholds, and merges the resulting labels back into the
original trip DataFrames.

Example:
    >>> labeler = TruckDataWeatherLabeler(cfg)
    >>> labeler.run()
"""

import logging
import time
from pathlib import Path
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
    """Labels electric truck trip data with weather and air temperature conditions.

    Segments verified trip files into time or distance windows, fetches
    hourly weather data from Meteostat for each window, classifies weather
    conditions using meteorological thresholds, and merges the labels back
    into the original trip DataFrame.

    Attributes:
        cfg (DictConfig): Hydra configuration object.  Must contain a ``data_preparation`` 
            sub-config with the fields``verified_data_dir``, ``file_extension``, ``labeled_data_dir``,
            ``results_dir``, ``window_size_km``, and ``window_size_seconds``.
        verified_data_dir (Path): Directory containing verified parquet trip files.
        file_extension (str): File suffix used when globbing trip files (e.g. ``'.parquet'``).
        labeled_data_dir (Path): Directory where labeled parquet files are written.
        results_dir (Path): Directory where label distribution JSON files are saved.
        window_size_km (float): Window size in kilometres for distance-based windowing.
        window_size_seconds (int): Window size in seconds for time-based windowing.
        weather_labeling_signals (list[str]): Signal columns used for weather labeling.
        weather_codes (dict[int, str]): Mapping from Meteostat condition codes to weather labels.
    """
    
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg

        self.verified_data_dir = Path(self.cfg.data_preparation.verified_data_dir) 
        self.file_extension = self.cfg.data_preparation.file_extension
        self.labeled_data_dir = Path(self.cfg.data_preparation.labeled_data_dir)
        self.results_dir =  Path(self.cfg.data_preparation.results_dir)
        self.window_size_km = self.cfg.data_preparation.window_size_km
        self.window_size_seconds = self.cfg.data_preparation.window_size_seconds

        self.weather_labeling_signals = const.WEATHER_LABELING_SIGNALS
        self.weather_codes = const.METEOSTAT_TO_WEATHER

        self.setup_dirs()
    
    def setup_dirs(self) -> None:
        """Creates required output directories if they do not already exist."""
        for path in [self.results_dir, self.labeled_data_dir]:
            path.mkdir(parents=True, exist_ok=True)

    def get_datasets(self) -> tuple[list[Path], list[str]]:
        """Gets all trip files in the verified data directory.

        Returns:
            tuple[list[Path], list[str]]: A tuple of ``(paths, filenames)`` where
                ``paths`` is a sorted list of :class:`pathlib.Path` objects
                matching ``file_extension``, and ``filenames`` contains the
                corresponding stems (no extension).
        """
        paths = sorted(self.verified_data_dir.glob(f"*{self.file_extension}"))
        filenames = [f.stem for f in paths]
        return paths, filenames
    
    def create_windows(self, 
        df: pd.DataFrame, 
        use_distance: bool = False, 
        use_time: bool = True
    ) -> pd.DataFrame:
        """Segments a trip DataFrame into fixed-size windows by distance or time.

        Assigns a ``window_id`` to each row and marks the first, middle, and
        last row of every window with boolean flags.

        Args:
            df (pd.DataFrame): Trip DataFrame to segment. Must contain
                ``hirestotalvehdist_cval_icuc`` if ``use_distance=True``.
            use_distance (bool, optional): If ``True``, windows are defined by
                ``window_size_km`` driven distance. Defaults to ``False``.
            use_time (bool, optional): If ``True``, windows are defined by
                ``window_size_seconds`` row count. Defaults to ``True``.

        Returns:
            pd.DataFrame: Input DataFrame with four additional columns:
                * ``window_id`` -- integer window index per row.
                * ``is_first_row`` -- ``True`` for the first row of each window.
                * ``is_middle_row`` -- ``True`` for the middle row of each window.
                * ``is_last_row`` -- ``True`` for the last row of each window.

        Raises:
            ValueError: If both or neither of ``use_distance`` and ``use_time``
                are ``True``.
        """
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
    
    @staticmethod
    def dew_point(temp: float, rhum: float) -> float:
        """Computes the dew point temperature from air temperature and relative humidity.

        Uses the Magnus formula as a fallback when dew point is not directly
        available from the weather station.

        Args:
            temp (float): Air temperature in degrees Celsius.
            rhum (float): Relative humidity as a percentage (0–100).

        Returns:
            float: Dew point temperature in degrees Celsius.

        References:
            .. admonition:: Papers

                Lawrence, M. G., 2005: The Relationship between Relative Humidity and the Dewpoint 
                Temperature in Moist Air: A Simple Conversion and Applications. 
                Bull. Amer. Meteor. Soc., 86, 225–234, https://doi.org/10.1175/BAMS-86-2-225. 

                Alduchov, O. A., and R. E. Eskridge, 1996: Improved Magnus Form Approximation of 
                Saturation Vapor Pressure. J. Appl. Meteor. Climatol., 35, 601–609, 
                https://journals.ametsoc.org/view/journals/apme/35/4/1520-0450_1996_035_0601_imfaos_2_0_co_2.xml. 
        """
        a, b = 17.27, 237.7
        alpha = (a * temp) / (b + temp) + np.log(rhum / 100.0)
        return (b * alpha) / (a - alpha)
  
    def fetch_hourly_weather_data(self, df: pd.DataFrame, time_delta_hour: int = 5) -> pd.DataFrame:
        """Fetches hourly weather data from Meteostat for each window in the trip.

        For each window, attempts to fetch weather data using the middle, first,
        or last row GPS point as a fallback chain. If all attempts fail, a row
        of ``NaN`` values is inserted. Missing temperature values are filled
        from the onboard sensor (``airtempoutsd_cval_cpc``) and missing dew
        points are computed via :meth:`dew_point`.

        Args:
            df (pd.DataFrame): Windowed trip DataFrame. Must contain
                ``signal_time``, ``window_id``, ``is_middle_row``,
                ``is_first_row``, ``is_last_row``, ``latitude_cval_ippc``,
                ``longitude_cval_ippc``, ``altitude_cval_ippc``, and
                ``airtempoutsd_cval_cpc`` columns.
            time_delta_hour (int, optional): Hours added/subtracted around the
                window timestamp to define the Meteostat query interval.
                Defaults to ``5``.

        Returns:
            pd.DataFrame: One row per window with the following columns:
                * ``window_id`` -- window index.
                * ``airtempoutsd_cval_cpc`` -- onboard air temperature sensor value.
                * ``temp`` (°C) -- air temperature.
                * ``dwpt`` (°C) -- dew point temperature.
                * ``rhum`` (%) -- relative humidity.
                * ``prcp`` (mm) -- one-hour precipitation total.
                * ``snow`` (mm) -- snow depth.
                * ``wdir`` (°) -- average wind direction.
                * ``wspd`` (km/h) -- average wind speed.
                * ``wpgt`` (km/h) -- peak wind gust.
                * ``pres`` (hPa) -- sea-level air pressure.
                * ``tsun`` (min) -- one-hour sunshine total.
                * ``coco`` (int) -- Meteostat weather condition code.
        Note:
            This function was developed with the assistance of Claude AI (Anthropic).
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
            weather_df.loc[mask, "dwpt"] = self.dew_point(
                weather_df.loc[mask, "temp"],
                weather_df.loc[mask, "rhum"]
            )

        return weather_df
      
    def label_weather_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Assigns weather and air temperature labels to each window using meteorological thresholds.

        Classification follows a priority chain:
        Meteostat code → storm → heavy snow → snow → heavy rain → rain →
        high wind → fog → clear. Small unlabeled gaps are filled by
        :meth:`fill_matching_gaps` when surrounding labels agree.

        Args:
            df (pd.DataFrame): Windowed weather DataFrame. Must contain
                ``airtempoutsd_cval_cpc``, ``dwpt``, ``rhum``, ``prcp``,
                ``snow``, ``wspd``, ``wpgt``, ``pres``, ``tsun``, and
                ``coco`` columns.

        Returns:
            pd.DataFrame: Input DataFrame with three additional columns:

            * ``air_temperature`` -- air temperature label (e.g. ``'cold'``, ``'warm'``).
            * ``weather`` -- weather condition label (e.g. ``'rain'``, ``'fog'``).
            * ``weather_gap_filled`` -- ``True`` where a gap was filled rather
            than directly observed.

        Note:
            This function was developed with the assistance of Claude AI (Anthropic).
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

        # AIR TEMPERATURE CLASSIFICATION
        air_temperature = pd.Series(index=df.index, dtype="string")
        air_temperature[temp < -10] = "extreme_cold"
        air_temperature[(temp >= -10) & (temp <= 0)] = "freezing_cold"
        air_temperature[(temp > 0) & (temp <= 10)] = "cold"
        air_temperature[(temp > 10) & (temp <= 20)] = "moderate"
        air_temperature[(temp > 20) & (temp <= 30)] = "warm"
        air_temperature[temp > 30] = "extreme_heat"

        # ATMOSPHERIC WEATHER CLASSIFICATION  
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
            (tsun < 15) &                  # < 15 min sunshine/hour
            (prcp < 0.5) &                 # Not raining
            (wspd < 15) &                  # Light wind (fog disperses in wind)
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
        
        # GAP FILLING
        weather = self.fill_matching_gaps(weather, max_gap_size=5)
        df["weather_gap_filled"] = weather.notna() & coco.map(self.weather_codes).isna()
        df["air_temperature"] = air_temperature
        df["weather"] = weather

        return df
    
    @staticmethod
    def fill_matching_gaps(series: pd.Series, max_gap_size: int = 3) -> pd.Series:
        """Fills small ``NaN`` gaps in a categorical Series by safe interpolation.

        A gap is filled only when both the preceding and following non-``NaN``
        values are identical, ensuring no label is invented at boundaries
        between different conditions.

        Args:
            series (pd.Series): Categorical Series with potential ``NaN`` gaps.
            max_gap_size (int, optional): Maximum number of consecutive ``NaN``
                values to fill. Gaps larger than this are left unchanged.
                Defaults to ``3``.

        Returns:
            pd.Series: Copy of ``series`` with qualifying gaps filled.
        """
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
    def label_truck_data(truck_data: pd.DataFrame, weather_data_labeled: pd.DataFrame) -> pd.DataFrame:
        """Merges weather labels into the original truck trip DataFrame.

        Joins ``air_temperature`` and ``weather`` columns from the labeled
        weather DataFrame onto the truck data using ``window_id`` as the key,
        then drops the temporary ``window_id`` and ``is_middle_row`` columns.

        Args:
            truck_data (pd.DataFrame): Original trip DataFrame containing a
                ``window_id`` column.
            weather_data_labeled (pd.DataFrame): Labeled weather DataFrame
                as returned by :meth:`label_weather_data`. Must contain
                ``window_id``, ``air_temperature``, and ``weather`` columns.

        Returns:
            pd.DataFrame: Truck DataFrame with ``air_temperature`` and
                ``weather`` columns added and ``window_id`` and
                ``is_middle_row`` dropped.
        """
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
    
    def save_label_distribution(self, df_labeled: pd.DataFrame, filename: str) -> None:
        """Computes and saves label distributions for weather columns as JSON.

        Calculates the percentage distribution of values in ``air_temperature``
        and ``weather`` columns and saves the result to ``results_dir``.

        Args:
            df_labeled (pd.DataFrame): Labeled trip DataFrame containing
                ``air_temperature`` and ``weather`` columns.
            filename (str): Output filename stem (without extension).
        """
        distribution = {}
        for col in ["air_temperature", "weather"]:
            if col in df_labeled.columns:
                dist = df_labeled[col].value_counts(normalize=True) * 100
                distribution[col] = dist.to_dict()
        
        save_path = self.results_dir / f"{filename}.json" 
        with open(save_path, "w") as f:
            json.dump(distribution, f, indent=4)

    def run(self) -> None:
        """Runs the full weather labeling pipeline over all verified trip files.

        For each trip file, the pipeline:

        1. Creates time-based windows with :meth:`create_windows`
        2. Fetches hourly weather data with :meth:`fetch_hourly_weather_data`
        3. Labels weather conditions with :meth:`label_weather_data`
        4. Merges labels into the trip data with :meth:`label_truck_data`
        5. Saves the labeled parquet and label distribution JSON

        Already-processed files are skipped. Logs total elapsed time on completion.
        """
        start_time = time.time()
        paths, filenames = self.get_datasets()
       
        with tqdm(total=len(filenames), desc="Preprocessing datasets") as pbar:
            for i, (path, filename) in enumerate(zip(paths, filenames)):
                output_path = Path(self.labeled_data_dir, f"{filename}{self.file_extension}")
                if output_path.exists():
                    tqdm.write(f"Skipping {filename}: already processed")
                    pbar.update(1)
                    continue

                df = pd.read_parquet(path)
                df_windows = self.create_windows(df)
                weather_data = self.fetch_hourly_weather_data(df_windows)
                weather_data_labeled = self.label_weather_data(weather_data)
                df_labeled = self.label_truck_data(df, weather_data_labeled)
                df_labeled.to_parquet(output_path)
                self.save_label_distribution(df_labeled, filename)

                assert len(df) == len(df_labeled)

                pbar.update(1)
            
        elapsed_total = time.time() - start_time
        minutes, seconds = divmod(elapsed_total, 60)
        logger.info(f"Completed processing all {len(paths)} datasets in {int(minutes)} min {int(seconds)} sec")
          
            
