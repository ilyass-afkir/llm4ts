"""Data Understanding Pipeline.

Provides a top-level pipeline that dispatches to specific data understanding
steps based on a Hydra configuration object.

Supported steps:
    - ``analysis``              -- analyzes trip locations, durations, distances, and payloads
    - ``quality_verification``  -- validates schema, temporal sampling, and generates trajectory plots

Example:
    >>> pipeline = DataUnderstandingPipeline(cfg)
    >>> pipeline.run()
"""

from src.data_understanding.modules.analysis import TruckDataAnalyzer
from src.data_understanding.modules.quality_verification import TruckDataQualityVerificator


class DataUnderstandingPipeline:
    """Top-level pipeline for dispatching data understanding tasks.

    Reads ``cfg.data_understanding.name`` to select and execute exactly one
    understanding stage.

    Attributes:
        cfg (DictConfig): Hydra configuration object. Must contain a ``data_understanding.name`` 
        field with one of the supported step names ``analysis`` or ``quality_verification`.
    """

    def __init__(self, cfg) -> None:
        self.cfg = cfg

    def run(self) -> None:
        """Executes the configured data understanding step.

        Dispatches to the corresponding module based on ``cfg.data_understanding.name``.

        Raises:
            ValueError: If ``cfg.data_understanding.name`` does not match any supported step.
        """
        
        if self.cfg.data_understanding.name == "analysis":
            analyzer = TruckDataAnalyzer(self.cfg)
            analyzer.run()

        if self.cfg.data_understanding.name == "quality_verification":
            verificator = TruckDataQualityVerificator(self.cfg)
            verificator.run()