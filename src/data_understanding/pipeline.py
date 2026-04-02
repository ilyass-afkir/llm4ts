"""
Description here.
"""

from src.data_understanding.modules.analysis import TruckDataAnalyzer
from src.data_understanding.modules.quality_verification import TruckDataQualityVerificator


class DataUnderstandingPipeline:
    def __init__(self, cfg):
        self.cfg = cfg

    def run(self):
        if self.cfg.data_understanding.name == "analysis":
            analyzer = TruckDataAnalyzer(self.cfg)
            analyzer.run()

        if self.cfg.data_understanding.name == "quality_verification":
            verificator = TruckDataQualityVerificator(self.cfg)
            verificator.run()