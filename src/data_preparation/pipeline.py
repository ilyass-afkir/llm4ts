"""Data preparation pipeline.

This module provides a pipeline that dispatches to specific
data preparation steps based on hydra configuration.

Supported steps:
    - weather_labeling
    - highway_labeling
    - preparation
"""

from src.data_preparation.modules.weather_labeling import TruckDataWeatherLabeler
from data_preparation.modules.highway_labeling import TruckDataHighwayLabeler
from src.data_preparation.modules.preparation import TruckDataPreparator


class DataPreparationPipeline:
    """Top-level pipeline for dispatching data preparation tasks.

    Reads ``cfg.data_preparation.name`` to select and execute exactly one
    preparation stage. Each stage is self-contained and produces outputs
    consumed by downstream pipelines.

    Attributes:
        cfg: Hydra configuration object containing all data preparation settings.

    Example:
        >>> pipeline = DataPreparationPipeline(cfg)
        >>> pipeline.run()
    """

    def __init__(self, cfg):
        """Initializes DataPreparationPipeline.

        Args:
            cfg: Hydra configuration object. Must contain a
                ``data_preparation.name`` field with one of the supported
                step names.
        """
        self.cfg = cfg
 
    def run(self):
        """Executes the configured data preparation step.

        Dispatches to the corresponding module based on
        ``cfg.data_preparation.name``.

        Raises:
            ValueError: If ``cfg.data_preparation.name`` does not match
                any supported step.
        """

        if self.cfg.data_preparation.name == "weather_labeling":
            weather_labeler = TruckDataWeatherLabeler(self.cfg)
            weather_labeler.run()
        elif self.cfg.data_preparation.name == "highway_labeling":
            highway_labeler = TruckDataHighwayLabeler(self.cfg)
            highway_labeler.run()
        elif self.cfg.data_preparation.name == "preparation":
            preparator = TruckDataPreparator(self.cfg)
            preparator.run()
        else:
            raise ValueError(
                f"Unknown data preparation step: '{self.cfg.data_preparation.name}'. "
                f"Supported steps: 'weather_labeling', 'highway_labeling', 'preparation'."
            )
     



        