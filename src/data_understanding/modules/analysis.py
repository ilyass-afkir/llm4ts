"""Electric Truck Data Analysis Module.
 
Provides analysis and visualization utilities for understanding electric truck
trip datasets, including trip durations, distances, payloads, geographic
locations, and vehicle group distributions.
 
Example:
    >>> analyzer = TruckDataAnalyzer(cfg)
    >>> analyzer.run()
"""

import logging
import json
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from omegaconf import DictConfig
from tqdm import tqdm
import reverse_geocoder as rg

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)


class TruckDataAnalyzer:
    """Analyzes electric truck trip datasets for data understanding purposes.
 
    Loads verified parquet trip files and computes statistics and visualizations
    covering trip locations, durations, distances, payloads, and vehicle group
    distributions. Results are saved as JSON summaries and publication-quality
    PDF/PNG plots.
 
    Attributes:
        cfg (DictConfig): Hydra configuration object. Must contain a 
            ``data_understanding`` sub-config with the fields
            ``file_extension``, ``verified_data_dir``, and ``results_dir``.
        verified_data_dir (Path): Directory containing verified parquet trip files.
        file_extension (str): File suffix used when globbing trip files (e.g. ``'.parquet'``).
        results_dir (Path): Directory where JSON summaries and plots are saved.
    """
    
    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        self.verified_data_dir = Path(self.cfg.data_understanding.verified_data_dir)
        self.file_extension = self.cfg.data_understanding.file_extension
        self.results_dir = Path(self.cfg.data_understanding.results_dir)
        self.setup_dirs()

    def setup_dirs(self) -> None:   
        """Creates required output directories if they do not already exist."""
        self.results_dir.mkdir(parents=True, exist_ok=True)

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
    
    @staticmethod
    def seconds_to_hms(seconds: float) -> str:
        """Converts a duration in seconds to an ``HH:MM:SS`` string.
 
        Args:
            seconds (float): Duration in seconds.
 
        Returns:
            str: Formatted duration string in ``HH:MM:SS`` format.
        
        Note:
            This function was developed with the assistance of ChatGPT.
        """
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        return f"{h:02d}:{m:02d}:{s:02d}"
    
    def get_trip_durations_and_distances(self) -> pd.DataFrame:
        """Computes average trip durations and distances grouped by vehicle group.
 
        Reads all verified parquet files, computes per-trip durations and 
        distances, and aggregates them by vehicle group (``v_group``).
 
        Returns:
            pd.DataFrame: Summary DataFrame with columns ``v_group``, 
            ``n_trips``, ``avg_duration`` (HH:MM:SS), and ``avg_distance_km``.
        """
        paths, filenames = self.get_datasets()

        for path, _ in tqdm(zip(paths, filenames), total=len(paths), desc="Calculating average trip durations and distances"):
            df = pd.read_parquet(path)
            df["signal_time"] = pd.to_datetime(df["signal_time"])

            trip_durations = (
                df.groupby(["v_group", "v_id"])["signal_time"]
                .apply(lambda x: (x.max() - x.min()).total_seconds())
                .reset_index(name="duration_s")
            )

            avg_duration = (
                trip_durations.groupby("v_group")["duration_s"]
                .mean()
                .apply(self.seconds_to_hms)
            )

            avg_distance = (
                df.groupby(["v_group", "v_id"])["hirestotalvehdist_cval_icuc"]
                .max()
                .groupby("v_group")
                .mean()
            )

            summary = pd.DataFrame({
                "n_trips": df.groupby("v_group")["v_id"].nunique(),
                "avg_duration": avg_duration,
                "avg_distance_km": avg_distance
            }).reset_index()

        return summary

    def get_trip_locations(self) -> dict[str, list[str]]:
        """Assigns country and region labels to all GPS points with reverse geocoding.
 
        Loads all verified parquet files, reverse-geocodes every GPS coordinate, 
        and groups trip filenames by their dominant country or country combination.
        Results are saved to ``results_dir/trips_by_country.json``.
 
        Returns:
            dict[str, list[str]]: Mapping from country key (e.g. ``'DE'``,
            ``'DE,FR'``) to a list of trip filenames belonging to that key.
        Note:
            This function was developed with the assistance of Claude AI (Anthropic).
        """
        paths, filenames = self.get_datasets()
    
        df = pl.concat([
            pl.scan_parquet(path).with_columns(pl.lit(fn).alias("filename"))
            for path, fn in tqdm(zip(paths, filenames), total=len(paths), desc="Loading files")
        ]).collect()

        coords = df.select(["latitude_cval_ippc", "longitude_cval_ippc"]).to_numpy()
        coords_list = [(lat, lon) for lat, lon in coords]
        results = rg.search(coords_list, mode=2)
        
        df = df.with_columns([
            pl.Series("country", [r['cc'] for r in results]),
            pl.Series("region", [r['admin1'] for r in results])
        ])

        file_country_counts = (
            df.group_by(["filename", "country"])
            .len()
            .rename({"len": "row_count"})
        )

        file_countries = (
            file_country_counts
            .group_by("filename")
            .agg([
                pl.col("country"),
                pl.col("row_count")
            ])
        )

        trips_dict = {}

        for fn, countries, counts in file_countries.iter_rows():
            valid_countries = [c for c, n in zip(countries, counts) if n >= 1000]

            if len(valid_countries) == 0:
                main_country = countries[counts.index(max(counts))]
                key = main_country
            elif len(valid_countries) == 1:
                key = valid_countries[0]
            else:
                key = ",".join(sorted(valid_countries))

            trips_dict.setdefault(key, []).append(fn)

        with open(self.results_dir / "trips_by_country.json", "w") as f:
            json.dump(trips_dict, f, indent=4)

        return trips_dict
        
    def generate_trips_summary(self) -> dict[str, list[str] | dict[str, any]]:
        """Generates a statistical summary of trips grouped by country.
 
        Reads ``trips_by_country.json`` and computes per-country trip counts,
        percentages, and multi-country breakdowns. Results are saved to
        ``results_dir/trips_summary.json``.
 
        Returns:
            dict[str, list[str] | dict[str, any]]: Summary dictionary containing country lists, per-key trip
                counts and percentages, and a ``multi_country_breakdown`` entry.
        Note:
            This function was developed with the assistance of Claude AI (Anthropic).
        """
        with open(self.results_dir / "trips_by_country.json", "r") as f:
            trips_dict = json.load(f)

        total_trips = sum(len(filenames) for filenames in trips_dict.values())
        single_country_keys = [key for key in trips_dict.keys() if "," not in key]
        single_countries = set(single_country_keys)
        all_countries = set()
        for key in trips_dict.keys():
            for c in key.split(","):
                all_countries.add(c)
        multi_countries = all_countries - single_countries

        # Sort lists
        all_countries = sorted(all_countries)
        single_countries = sorted(single_countries)
        multi_countries = sorted(multi_countries)

        summary = {
            "all_countries": all_countries,
            "single_countries": single_countries,
            "multi_countries": multi_countries
        }

        # Per-key trips and percentages
        for key, files in trips_dict.items():
            summary[key] = {
                "num_trips": len(files),
                "pct_total": round(len(files) / total_trips * 100, 2)
            }

        multi_country_files = [f for k, fs in trips_dict.items() if "," in k for f in fs]
        multi_country_count = len(multi_country_files)
        multi_country_pct = round(multi_country_count / total_trips * 100, 2)

        # Count of each country in multi-country trips
        multi_country_countries = {}
        for key, files in trips_dict.items():
            if "," in key:
                for c in key.split(","):
                    multi_country_countries[c] = multi_country_countries.get(c, 0) + len(files)

        multi_country_countries_pct = {
            c: round(count / multi_country_count * 100, 2)
            for c, count in multi_country_countries.items()
        }

        summary["multi_country_breakdown"] = {
            "total_files": multi_country_count,
            "pct_of_total": multi_country_pct,
            "countries_breakdown": {
                c: {"num_files": multi_country_countries[c], "pct": multi_country_countries_pct[c]}
                for c in multi_country_countries
            }
        }

        with open(self.results_dir / "trips_summary.json", "w") as f:
            json.dump(summary, f, indent=4)

        return summary
    
    def plot_trips_summary(self) -> None:
        """Plots and saves a vertical bar chart of trip distribution by country.
 
        Reads ``trips_by_country.json``, separates single-country and
        multi-country trips, and produces a log-scale bar chart with count and
        percentage annotations. Saves both as PDF and PNG files to ``results_dir``.
        """
        json_path = self.results_dir / "trips_by_country.json"
        with open(json_path, "r") as f:
            trips_dict = json.load(f)

        country_names = {
            'AT': 'Austria', 'BG': 'Bulgaria', 'DE': 'Germany', 'DK': 'Denmark',
            'ES': 'Spain', 'FI': 'Finland', 'FR': 'France', 'HR': 'Croatia',
            'IT': 'Italy', 'NL': 'Netherlands', 'RS': 'Serbia', 'SE': 'Sweden',
            'SI': 'Slovenia', 'TR': 'Turkey'
        }

        # Separate single and multiple countries
        single_country_data = []
        multi_countries_count = 0
        total_trips = sum(len(v) for v in trips_dict.values())

        for country_key, trips in trips_dict.items():
            num_trips = len(trips)
            pct_total = (num_trips / total_trips) * 100
            
            if ',' in country_key:
                # Multiple countries
                multi_countries_count += num_trips
            else:
                # Single country
                single_country_data.append({
                    "country": country_key,
                    "num_trips": num_trips,
                    "pct_total": pct_total
                })
        
        multi_countries_pct = (multi_countries_count / total_trips) * 100

        # Verify percentages sum to 100%
        total_pct = sum([c["pct_total"] for c in single_country_data]) + multi_countries_pct
        logger.info(f"Total percentage: {total_pct:.3f}%")
        assert abs(total_pct - 100.0) < 0.01, f"Percentages don't sum to 100%: {total_pct}"

        # Sort single countries by trips (descending)
        single_country_data.sort(key=lambda x: x["num_trips"], reverse=True)

        # Plot settings
        font_path_normal = Path("llm-erange/src/utils/times.ttf")
        font_path_bold = Path("llm-erange/src/utils/times_bold.ttf")
        fm.fontManager.addfont(str(font_path_normal))
        fm.fontManager.addfont(str(font_path_bold))
        prop_normal = fm.FontProperties(fname=str(font_path_normal))
        plt.rcParams["font.family"] = prop_normal.get_name()
        plt.rcParams["font.size"] = 12

        # Combine single and multiple countries
        countries_full = [country_names.get(c["country"], c["country"]) for c in single_country_data]
        countries_full.append("Two countries")
        
        num_trips = [c["num_trips"] for c in single_country_data]
        num_trips.append(multi_countries_count)
        
        pct_total = [c["pct_total"] for c in single_country_data]
        pct_total.append(multi_countries_pct)

        fig, ax = plt.subplots(figsize=(18, 8), dpi=300)
        
        # Styling
        ax.set_facecolor('white')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#000000')
        ax.spines['bottom'].set_color('#000000')
        ax.spines['left'].set_linewidth(0.5)
        ax.spines['bottom'].set_linewidth(0.5)
        ax.grid(True, alpha=0.15, linestyle='-', linewidth=0.6, color="#FFFFFF", axis='y')
        ax.set_axisbelow(True)
        ax.tick_params(axis='both', colors='#000000', labelsize=12)

        # Calculate average for color comparison
        avg_trips = np.mean(num_trips)
        colors = ['#4d4943' if trips >= avg_trips else '#f5f5f5' for trips in num_trips]

        x = np.arange(len(countries_full))
        bars = ax.bar(x, num_trips, color=colors, edgecolor='black', linewidth=0.5, alpha=1)

        # Add count and percentage labels on top of bars (2 lines)
        for i, (bar, count, pct) in enumerate(zip(bars, num_trips, pct_total)):
            y_pos = bar.get_height() * 1.4
            # Count line (bold)
            ax.text(bar.get_x() + bar.get_width()/2, y_pos, 
                    f'{count}', 
                    ha='center', va='bottom', fontsize=12, color='#000000', weight='normal')
            # Percentage line (normal weight) - more space
            ax.text(bar.get_x() + bar.get_width()/2, y_pos * 0.77, 
                    f'({pct:.3f}%)', 
                    ha='center', va='bottom', fontsize=12, color='#000000')
    
        # Use log scale for y-axis to better visualize all bars
        ax.set_yscale('log')

        # Add average line
        ax.axhline(y=avg_trips, color='#FDCA00', linestyle='--', linewidth=2, alpha=1, 
                label=f'Average: {round(avg_trips)}')
        legend = ax.legend(loc='upper right', fontsize=10, framealpha=1)
        legend.get_frame().set_edgecolor('black')
        legend.get_frame().set_linewidth(0.5)

        ax.set_ylabel("Number of Trips", fontsize=12, labelpad=12, color='#000000', fontweight="normal")
        ax.set_xlabel("Country", fontsize=12, labelpad=12, color='#000000', fontweight='normal')
        ax.set_xticks(x)
        ax.set_xticklabels(countries_full, rotation=0, ha='center')
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{int(y)}'))
        fig.patch.set_facecolor('white')
        plt.tight_layout(pad=2.0)

        # Save both PDF and PNG
        png_path = self.results_dir / "trips_summary_plot.png"
        pdf_path = self.results_dir / "trips_summary_plot.pdf"
        plt.savefig(pdf_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        logger.info(f"Saved plot to {png_path} and {pdf_path}")

    def plot_v_groups(self) -> None:
        """Plots and saves a vertical bar chart of trip counts per vehicle group.
 
        Reads all verified parquet files, counts trips per vehicle group
        (``v_group``), and produces a log-scale bar chart with count and
        percentage annotations. Saves both as PDF and PNG files to ``results_dir``.
        """
        paths, filenames = self.get_datasets()
        v_group_counts = {}
        for path, _ in tqdm(zip(paths, filenames), total=len(paths), desc="Ploting v_group counts"):
            df = pd.read_parquet(path)
            key = df["v_group"].iloc[0]
            v_group_counts[key] = v_group_counts.get(key, 0) + 1

        # Sort by v_group number (ascending)
        sorted_items = sorted(v_group_counts.items(), key=lambda x: int(x[0]) if str(x[0]).isdigit() else float('inf'))
        groups = [str(item[0]) for item in sorted_items]
        counts = [item[1] for item in sorted_items]
        
        total_counts = sum(counts)
        pct_total = [(c / total_counts) * 100 for c in counts]

        # Verify percentages sum to 100%
        total_pct = sum(pct_total)
        logger.info(f"Total percentage: {total_pct:.3f}%")
        assert abs(total_pct - 100.0) < 0.01, f"Percentages don't sum to 100%: {total_pct}"

        font_path_normal = Path("llm-erange/src/utils/times.ttf")
        font_path_bold = Path("llm-erange/src/utils/times_bold.ttf")
        fm.fontManager.addfont(str(font_path_normal))
        fm.fontManager.addfont(str(font_path_bold))
        prop_normal = fm.FontProperties(fname=str(font_path_normal))
        plt.rcParams["font.family"] = prop_normal.get_name()
        plt.rcParams["font.size"] = 12

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8), dpi=300)

        # Styling
        ax.set_facecolor('white')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color("#000000")
        ax.spines['bottom'].set_color("#000000")
        ax.spines['left'].set_linewidth(0.5)
        ax.spines['bottom'].set_linewidth(0.5)
        ax.grid(True, alpha=0.15, linestyle='-', linewidth=0.6, color="#FFFFFF", axis='y')
        ax.set_axisbelow(True)
        ax.tick_params(axis='both', colors="#000000", labelsize=12)

        # Color bars based on average
        avg_count = np.mean(counts)
        colors = ['#4d4943' if c >= avg_count else '#f5f5f5' for c in counts]

        # Vertical bar plot
        x = np.arange(len(groups))
        bars = ax.bar(x, counts, width=0.8, color=colors, edgecolor='black', linewidth=0.5, alpha=1)

        # Add count and percentage labels
        for i, (bar, count, pct) in enumerate(zip(bars, counts, pct_total)):
            y_pos = bar.get_height() * 1.2
            ax.text(bar.get_x() + bar.get_width()/2, y_pos, f'{count}', ha='center', va='bottom', fontsize=12, color='#000000', weight='normal')
            ax.text(bar.get_x() + bar.get_width()/2, y_pos * 0.85, f'({pct:.3f}%)', ha='center', va='bottom', fontsize=12, color='#000000')

        # Log scale
        ax.set_yscale('log')

        # Average line
        ax.axhline(y=avg_count, color='#FDCA00', linestyle='--', linewidth=2, alpha=1, label=f'Average: {round(avg_count)}')
        legend = ax.legend(loc='upper right', fontsize=10, framealpha=1)
        legend.get_frame().set_edgecolor('black')
        legend.get_frame().set_linewidth(0.5)

        # Labels and x-axis
        ax.set_ylabel("Number of Trips", fontsize=12, labelpad=12, color='#000000', fontweight="normal")
        ax.set_xlabel("Vehicle Group", fontsize=12, labelpad=12, color='#000000', fontweight='normal')
        ax.set_xticks(x)
        ax.set_xticklabels(groups, rotation=0, ha='center')

        # Format y-axis ticks
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{int(y)}'))
        fig.patch.set_facecolor('white')

        plt.tight_layout(pad=2.0)

        # Save plots
        png_path = self.results_dir / "v_group_counts_plot.png"
        pdf_path = self.results_dir / "v_group_counts_plot.pdf"
        plt.savefig(pdf_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        logger.info(f"Saved plot to {png_path} and {pdf_path}")
     
    def collect_trip_metadata(self) -> dict[str, int | float | list[float]]:
        """Collects and saves descriptive statistics for trip metadata.
 
        Iterates over all verified parquet files and aggregates statistics
        for payload, distance, and duration. Results are saved to
        ``results_dir/meta_data.json``.
 
        Returns:
            dict[str, int | float | list[float]]: Metadata dictionary containing ``total_trips``, 
            min/max/mean/std and sequence values for ``payload``, ``distance``, and ``duration``.
        """
        paths, _ = self.get_datasets()

        meta = {
            "total_trips": len(paths),
            "payloads": set(),
            "distances": [],
            "durations": []
        }

        for path in tqdm(paths, total=len(paths), desc="collecting"):
            df = pd.read_parquet(path)
            meta["payloads"].add(df["vehweight_cval_pt"].iloc[0])
            meta["distances"].append(df["hirestotalvehdist_cval_icuc"].iloc[-1])
            t = pd.to_datetime(df["signal_time"])
            dur_min = (t.iloc[-1] - t.iloc[0]).total_seconds() / 60
            meta["durations"].append(dur_min)

        payload_list = list(meta["payloads"])
        meta["min_payload"] = min(payload_list) 
        meta["max_payload"] = max(payload_list) 
        meta["mean_payload"] = float(np.mean(payload_list))
        meta["std_payload"] = float(np.std(payload_list))
        meta["payload_sequence"] = [float(x) for x in payload_list]

        d = meta["distances"]
        meta["min_distance"] = min(d)
        meta["max_distance"] = max(d)
        meta["mean_distance"] = float(np.mean(d))
        meta["std_distance"] = float(np.std(d))
        meta["distance_sequence"] = [float(x) for x in d]

        u = meta["durations"]
        meta["min_duration"] = min(u) 
        meta["max_duration"] = max(u)
        meta["mean_duration"] = float(np.mean(u))
        meta["std_duration"] = float(np.std(u))
        meta["duration_sequence"] = [float(x) for x in u]

        # Remove temporary raw lists and set
        del meta["payloads"]
        del meta["distances"]
        del meta["durations"]

        with open(self.results_dir / "meta_data.json", "w") as f:
            json.dump(meta, f, indent=4)

        return meta
    
    def plot_trip_metadata(self) -> None:
        """Plots and saves a boxplot of trip distance, duration, and payload.
 
        Reads ``meta_data.json`` and produces a side-by-side boxplot for
        distance (km), duration (min), and payload (t). Saves both PDF and
        PNG to ``results_dir``.
        """
        json_path = self.results_dir / "meta_data.json"
        with open(json_path, "r") as f:
            meta = json.load(f)
        
        # Fonts
        font_path_normal = Path("llm-erange/src/utils/times.ttf")
        font_path_bold = Path("llm-erange/src/utils/times_bold.ttf")
        fm.fontManager.addfont(str(font_path_normal))
        fm.fontManager.addfont(str(font_path_bold))
        prop_normal = fm.FontProperties(fname=str(font_path_normal))
        plt.rcParams["font.family"] = prop_normal.get_name()
        plt.rcParams["font.size"] = 12

        # Extract sequences
        payload_seq = meta["payload_sequence"]
        distance_seq = meta["distance_sequence"]
        duration_seq = meta["duration_sequence"]
        
        box_data = [distance_seq, duration_seq, payload_seq]
        labels = ["Distance", "Duration", "Payload"]

        fig, ax = plt.subplots(figsize=(11, 7), dpi=300)

        # Styling
        ax.set_facecolor("white")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#000000")
        ax.spines["bottom"].set_color("#000000")
        ax.spines["left"].set_linewidth(0.5)
        ax.spines["bottom"].set_linewidth(0.5)
        ax.grid(True, alpha=0.15, linestyle="-", linewidth=0.6, color="#FFFFFF", axis="y")
        ax.set_axisbelow(True)
        ax.tick_params(axis="both", colors="#000000", labelsize=12)

        # Boxplot colors
        box_fill = "#ffe06c"
        median_color = "#000000"
        black_line = "#000000"

        # Create boxplot with actual data sequences
        bp = ax.boxplot(
            box_data,
            labels=labels,
            widths=0.6,
            patch_artist=True,
            boxprops=dict(facecolor=box_fill, edgecolor=black_line, linewidth=0.5),
            whiskerprops=dict(color=black_line, linewidth=0.5),
            capprops=dict(color=black_line, linewidth=0.5),
            medianprops=dict(color=median_color, linewidth=1.5),
            flierprops=dict(marker='o', markerfacecolor='#F5F5F5', markeredgecolor='#000000', 
                        markersize=5, markeredgewidth=0.5),
        )

        ax.set_ylabel("", fontsize=12, fontweight="bold", color="#000000")
        ax.set_xlabel("", fontsize=12)

        # Update x-axis labels with units
        ax.set_xticklabels(["Distance (km)", "Duration (min)", "Payload (t)"])
        fig.patch.set_facecolor("white")
        plt.tight_layout(pad=2.0)

        png_path = self.results_dir / "meta_data_boxplot.png"
        pdf_path = self.results_dir / "meta_data_boxplot.pdf"
        plt.savefig(pdf_path, dpi=300, bbox_inches="tight", facecolor="white")
        plt.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close()
        logger.info(f"Saved plot: {png_path}")

    def run(self) -> None:
        """Runs the full data understanding pipeline in sequence.
 
        Executes all analysis and visualization steps:
 
        1. Assigns country labels to trip GPS points
        2. Generates a country-level trip summary
        3. Plots trip distribution by country
        4. Plots trip counts by vehicle group
        5. Collects trip metadata statistics
        6. Plots trip metadata boxplots
        """
        self.get_trip_locations()
        self.generate_trips_summary()
        self.plot_trips_summary()
        self.plot_v_groups()
        self.collect_trip_metadata()
        self.plot_trip_metadata()
     




