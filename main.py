"""Main entry point for the ML pipeline orchestration.

This module provides the top-level execution logic for sequentially running
the data understanding, data preparation, and training/evaluation pipelines.
Pipeline execution is controlled via Hydra configuration flags.

Example using hydra:
    >>> python llm-erange/main.py +experiment/highway=100_8B
"""

import logging
import hydra
from omegaconf import DictConfig

from src.data_understanding.pipeline import DataUnderstandingPipeline
from src.data_preparation.pipeline import DataPreparationPipeline
from src.training_and_evaluation.pipeline import TrainingEvaluationPipeline

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

@hydra.main(config_path="configs", config_name="main_config", version_base=None)
def main(cfg: DictConfig):
    """Entry point for running the ML pipelines based on configuration flags.

    Sequentially executes the data understanding, data preparation, and
    training/evaluation pipelines depending on which flags are enabled in the
    Hydra configuration. Within the training pipeline, exactly one execution
    mode must be selected: UEA benchmark suite, truck data, or custom
    benchmarks.

    Args:
        cfg: Hydra configuration object loaded from ``configs/main_config``.
            Expected top-level keys:

            * ``run_flags.data_understanding`` (bool): Whether to run the data
              understanding pipeline.
            - ``run_flags.data_preparation`` (bool): Whether to run the data
              preparation pipeline.
            - ``run_flags.run_training`` (bool): Whether to run the training
              and evaluation pipeline.
            - ``training.use_uea`` (bool): Run UEA benchmark suite and plot
              critical-difference diagrams for all metrics.
            - ``training.use_truck_data`` (bool): Run pipeline on truck
              dataset.
            - ``training.use_benchmark`` (bool): Run custom benchmark
              experiments.

    Returns:
        None
    """
    if cfg.run_flags.data_understanding:
        logger.info("Running data understanding pipeline")
        pipeline = DataUnderstandingPipeline(cfg)
        pipeline.run()


    if cfg.run_flags.data_preparation:
        logger.info("Running data preparation pipeline")
        pipeline = DataPreparationPipeline(cfg)
        pipeline.run()

    if cfg.run_flags.run_training:
        logger.info("Running training and evalaution pipeline")
        pipeline = TrainingEvaluationPipeline(cfg)
        if cfg.training.use_uea:
            pipeline.run_uea()
            pipeline.plot_cd_all_metrics()
        elif cfg.training.use_truck_data:
            pipeline.run_truck_data()
        elif cfg.training.use_benchmark:
            pipeline.run_benchmarks()
        else:
            logger.warning("Select an option UEA or truck data or benchmark")

if __name__ == "__main__":
    main()
