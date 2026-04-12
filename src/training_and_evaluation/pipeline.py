"""Training and evaluation pipeline for time-series classification models.

This module orchestrates end-to-end training and evaluation across multiple
model architectures and dataset types, including UEA benchmark datasets and
proprietary truck sensor data. It supports baseline comparisons via ROCKET
and InceptionTime classifiers, few-shot training regimes, and critical-
difference diagram generation for statistical model comparison.

Example:
    >>> pipeline = TrainingEvaluationPipeline(cfg)
    >>>
    >>> # Run on UEA benchmark datasets
    >>> pipeline.run_uea()
    >>>
    >>> # Run on truck sensor data (full + few-shot)
    >>> pipeline.run_truck_data()
    >>>
    >>> # Run classical baseline benchmarks
    >>> pipeline.run_benchmarks()
    >>>
    >>> # Generate critical-difference plots across all metrics
    >>> TrainingEvaluationPipeline.plot_cd_all_metrics()
"""

from omegaconf import DictConfig, OmegaConf
from pathlib import Path
import torch
import gc
import json
import logging
import time

from sktime.classification.deep_learning import InceptionTimeClassifier
from sktime.classification.kernel_based import RocketClassifier
import numpy as np
from aeon.visualisation import plot_critical_difference
from matplotlib import font_manager as fm
import matplotlib.pyplot as plt

from src.data_preparation.modules.constants import UEA
from src.data_preparation.modules.truck_data_loaders import TruckDataFactory, TruckDataFactoryBenchmark
from src.data_preparation.modules.uea_data_loaders import UEADataFactory
from src.models.one_fits_all import OneFitsAll
from src.models.llm_few import LLMFew
from src.models.time_llm import TimeLLM
from src.models.deep_range_v1 import DeepRange
from src.models.deep_range_v2 import DeepRangeV2
from src.training_and_evaluation.modules.trainer import Trainer
from src.training_and_evaluation.modules.evaluator import UEAEvaluator, TruckEvaluator, TruckEvaluatorBenchmark
from src.utils.model_summarizer import ModelSummarizer

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)


class TrainingEvaluationPipeline:
    """End-to-end pipeline for training and evaluating time-series classifiers.

    Manages directory setup, model instantiation, training, evaluation, and
    result serialization for multiple experimental configurations. Supports
    UEA benchmark datasets, truck sensor data, and classical baseline models.

    Attributes:
        cfg (DictConfig): Hydra configuration object containing at
                minimum ``cfg.training.save_results_dir_path``,
                ``cfg.training.save_results_inception``,
                ``cfg.training.save_results_rocket``, and
                ``cfg.training.save_results_resnet``.
    """

    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg

        self.paths = {
            "main": Path(self.cfg.training.save_results_dir_path),
            "main_fewshot": Path(self.cfg.training.save_results_dir_path) / "fewshot",

            "inception": Path(self.cfg.training.save_results_inception),
            "inception_fewshot": Path(self.cfg.training.save_results_inception) / "fewshot",

            "rocket": Path(self.cfg.training.save_results_rocket),
            "rocket_fewshot": Path(self.cfg.training.save_results_rocket) / "fewshot",

            "resnet": Path(self.cfg.training.save_results_resnet),
            "resnet_fewshot": Path(self.cfg.training.save_results_resnet) / "fewshot",
        }

        self.setup_dirs()

    def setup_dirs(self) -> None:
        """Creates all output directories defined in ``self.paths``.

        Iterates over every path in ``self.paths`` and calls
        ``Path.mkdir(parents=True, exist_ok=True)``, so missing intermediate
        directories are created and existing ones are silently ignored.
        """
        for path in self.paths.values():
            path.mkdir(parents=True, exist_ok=True)

    def run_uea(self) -> None:
        """Trains and evaluates the configured model on all UEA datasets.

        Iterates over every dataset defined in the ``UEA`` registry. For each
        dataset the configuration is updated in-place with dataset-specific
        hyperparameters (channels, classes, sequence length, batch size, patch
        stride, and patch length), a data loader pair is built, and the model
        is trained and evaluated. Per-dataset training times and evaluation
        metrics are persisted to disk after all datasets have been processed.

        The following files are written to ``self.paths["main"]``:

        * ``training_times.json`` – individual and total training times in
          seconds.
        * ``evaluation_report.json`` – aggregated evaluation metrics produced
          by :class:`UEAEvaluator`.
        * Model-summary files produced by :class:`ModelSummarizer`.

        Raises:
            ValueError: If ``cfg.model.name`` does not match any supported
                model identifier.
        """
        model_summarizer = ModelSummarizer(self.cfg, self.paths["main"])
        evaluator = UEAEvaluator(self.cfg)
        training_times: dict[str, float] = {}

        def run_single_dataset(name: str, train_loader: torch.utils.data.DataLoader, test_loader: torch.utils.data.DataLoader) -> None:
            """Trains and evaluates the model on a single UEA dataset.

            Args:
                name (str): Dataset name used as a key in evaluation reports
                    and training-time logs.
                train_loader (torch.utils.data.DataLoader): DataLoader yielding
                    ``(inputs, labels)`` batches for training.
                test_loader (torch.utils.data.DataLoader): DataLoader yielding
                    ``(inputs, labels)`` batches for evaluation.

            Raises:
                ValueError: If ``cfg.model.name`` is not one of
                    ``"one_fits_all"``, ``"llm_few"``, ``"time_llm"``,
                    ``"deep_range"``, or ``"deep_range_v2"``.
            """
            if self.cfg.model.name == "one_fits_all":
                model = OneFitsAll(self.cfg)
            elif self.cfg.model.name == "llm_few":
                model = LLMFew(self.cfg)
            elif self.cfg.model.name == "time_llm":
                model = TimeLLM(self.cfg)
            elif self.cfg.model.name == "deep_range":
                model = DeepRange(self.cfg)
            elif self.cfg.model.name == "deep_range_v2":
                model = DeepRangeV2(self.cfg)
            else:
                raise ValueError(f"Unknown model: {self.cfg.model.name}")

            _ = model_summarizer.summarize_model_parameters(model, name)

            trainer = Trainer(
                cfg=self.cfg,
                model=model,
                train_loader=train_loader,
                val_loader=None,
                save_dir=self.paths["main"]
            )

            model, elapsed = trainer.fit()
            training_times[name] = elapsed

            evaluator.evaluate_model(model=model, test_loader=test_loader, dataset_name=name)

            self.cleanup_objects([model.llm, model, trainer, train_loader, test_loader])

        for name, info in UEA.items():
            OmegaConf.update(self.cfg, "training.num_channels", info["num_channels"])
            OmegaConf.update(self.cfg, "training.num_classes", info["num_classes"])
            OmegaConf.update(self.cfg, "training.sequence_length", info["sequence_length"])
            OmegaConf.update(self.cfg, "training.batch_size", info["batch_size"])
            OmegaConf.update(self.cfg, "model.patch_stride", info["patch_stride"])
            OmegaConf.update(self.cfg, "model.patch_length", info["patch_length"])

            uea_factory = UEADataFactory(
                name,
                Path(self.cfg.training.uea_data_dir_path),
                self.cfg.training.batch_size,
                self.cfg.training.num_workers
            )

            train_loader, test_loader = uea_factory.build_train_test_loaders()

            if train_loader is None or test_loader is None:
                logger.warning(f"Skipping dataset {name} due to loading issues")
                continue

            run_single_dataset(name, train_loader, test_loader)

        _ = model_summarizer.average_model_summaries()

        training_times_output = {
            "individual_training_times": training_times,
            "total_training_time": sum(training_times.values())
        }

        with open(self.paths["main"] / "training_times.json", "w") as f:
            json.dump(training_times_output, f, indent=4)

        evaluator.save_evaluation_report()

    def run_truck_data(self) -> None:
        """Trains and evaluates DeepRangeV2 on the truck sensor dataset.

        Executes two training runs back-to-back:

        1. **Full training** – uses the complete training split.
        2. **Few-shot training** – uses the reduced few-shot split.

        Both runs share the same validation and test loaders. The best
        checkpoint from each run is loaded before evaluation. Results are
        written to ``self.paths["main"]`` and ``self.paths["main_fewshot"]``
        respectively by :class:`TruckEvaluator`.

        GPU memory is freed between training runs via
        :meth:`cleanup_objects`.
        """
        truck_factory = TruckDataFactory(
            batch_size=self.cfg.training.batch_size,
            num_workers=self.cfg.training.num_workers,
            data_path=self.cfg.training.truck_data_dir_path,
            window_size=self.cfg.training.sequence_length,
            label_col=self.cfg.training.label
        )

        train_loader, fewshot_train_loader, val_loader, test_loader = \
            truck_factory.build_train_val_test_loaders()

        # ===== normal training =====
        model = DeepRangeV2(self.cfg)
        trainer = Trainer(self.cfg, model, train_loader, val_loader, self.paths["main"])
        trainer.fit()

        evaluator = TruckEvaluator(self.cfg, model, test_loader, self.paths["main"])
        evaluator.load_best_model()
        evaluator.evaluate()

        self.cleanup_objects([trainer, evaluator, model])

        # ===== fewshot training =====
        model_fewshot = DeepRangeV2(self.cfg)
        trainer_fewshot = Trainer(self.cfg, model_fewshot, fewshot_train_loader, val_loader, self.paths["main_fewshot"])
        trainer_fewshot.fit()

        evaluator_fewshot = TruckEvaluator(self.cfg, model_fewshot, test_loader, self.paths["main_fewshot"])
        evaluator_fewshot.load_best_model()
        evaluator_fewshot.evaluate()

        self.cleanup_objects([model_fewshot, trainer_fewshot, evaluator_fewshot,
                               train_loader, fewshot_train_loader, val_loader, test_loader])

    def run_benchmarks(self) -> None:
        """Trains and evaluates classical baseline classifiers on truck data.

        Fits the following scikit-learn-compatible classifiers using both the
        full training set and the few-shot subset:

        * **MiniROCKET** (:class:`RocketClassifier` with 5 000 kernels).
        * **InceptionTime** (:class:`InceptionTimeClassifier`).

        Training wall-clock times (in minutes) for each variant are aggregated
        and written to
        ``self.paths["main"] / "training_times_benchmarks.json"``.
        Evaluation artefacts for each variant are written to the corresponding
        subdirectory in ``self.paths``.

        Note:
            This method does **not** train deep-learning models defined in
            ``src.models``; use :meth:`run_truck_data` for those.
        """
        data = TruckDataFactoryBenchmark(
            self.cfg.training.truck_data_dir_path,
            self.cfg.training.sequence_length,
            self.cfg.training.label
        )

        X_train, y_train, X_fewshot_train, y_fewshot_train, X_test, y_test = data.get_splits()

        training_times: dict[str, float] = {}

        # RocketClassifier (full data)
        rocket = RocketClassifier(
            num_kernels=5000,
            rocket_transform="minirocket",
            n_jobs=-1,
            use_multivariate="yes"
        )
        logger.info("Starting Rocket")
        start = time.time()
        rocket.fit(X_train, y_train)
        training_times["rocket_full"] = (time.time() - start) / 60
        evaluator = TruckEvaluatorBenchmark(self.cfg, rocket, X_test, y_test, self.paths["rocket"])
        evaluator.evaluate()
        logger.info("End Rocket")
        self.cleanup_objects([rocket, evaluator])

        # RocketClassifier (few-shot)
        rocket_fewshot = RocketClassifier(
            num_kernels=5000,
            rocket_transform="minirocket",
            n_jobs=-1,
            use_multivariate="yes"
        )
        logger.info("Starting Rocket")
        start = time.time()
        rocket_fewshot.fit(X_fewshot_train, y_fewshot_train)
        training_times["rocket_fewshot"] = (time.time() - start) / 60
        evaluator = TruckEvaluatorBenchmark(self.cfg, rocket_fewshot, X_test, y_test, self.paths["rocket_fewshot"])
        evaluator.evaluate()
        logger.info("End Rocket")
        self.cleanup_objects([rocket_fewshot, evaluator])

        # InceptionTimeClassifier (full data)
        inception = InceptionTimeClassifier(
            batch_size=self.cfg.training.batch_size,
            n_epochs=self.cfg.training.epochs,
            verbose=True
        )
        start = time.time()
        inception.fit(X_train, y_train)
        training_times["inception_full"] = (time.time() - start) / 60
        evaluator = TruckEvaluatorBenchmark(self.cfg, inception, X_test, y_test, self.paths["inception"])
        evaluator.evaluate()
        self.cleanup_objects([inception, evaluator])

        # InceptionTimeClassifier (few-shot)
        inception_fewshot = InceptionTimeClassifier(
            batch_size=self.cfg.training.batch_size,
            n_epochs=self.cfg.training.epochs,
            verbose=True
        )
        start = time.time()
        inception_fewshot.fit(X_fewshot_train, y_fewshot_train)
        training_times["inception_fewshot"] = (time.time() - start) / 60
        evaluator = TruckEvaluatorBenchmark(self.cfg, inception_fewshot, X_test, y_test, self.paths["inception_fewshot"])
        evaluator.evaluate()
        self.cleanup_objects([inception_fewshot, evaluator,
                               X_train, y_train, X_fewshot_train, y_fewshot_train, X_test, y_test])

        with open(self.paths["main"] / "training_times_benchmarks.json", "w") as f:
            json.dump(training_times, f, indent=4)

    @staticmethod
    def plot_cd_all_metrics() -> None:
        """Generates critical-difference diagrams for all evaluation metrics.

        Loads per-dataset evaluation reports from six pre-defined JSON files
        (One Fits All, Time-LLM, LLM-Few, S²IP-TEMPO, Deep Range V1, and
        Deep Range V2), computes the intersection of datasets common to all
        models, and produces a critical-difference diagram for each of the
        following metrics:

        * ``accuracy``
        * ``f1_score``
        * ``b_accuracy`` (balanced accuracy)

        Output files (PNG at 300 dpi and PDF) are written to the ``base_path``
        directory, named ``comp_uea_{metric}_with_dp.{ext}``. Pairwise
        p-values for all metric comparisons are consolidated and saved to
        ``p_values_with_dp.json``.

        Note:
            This is a **static** method; it reads fixed paths and does not
            depend on instance state. Font files are expected at
            ``llm-erange/src/utils/times.ttf`` and
            ``llm-erange/src/utils/times_bold.ttf`` relative to the working
            directory.
        """
        font_path_normal = "llm-erange/src/utils/times.ttf"
        font_path_bold = "llm-erange/src/utils/times_bold.ttf"
        fm.fontManager.addfont(font_path_normal)
        fm.fontManager.addfont(font_path_bold)
        prop_normal = fm.FontProperties(fname=font_path_normal)
        plt.rcParams["font.family"] = prop_normal.get_name()
        plt.rcParams["font.size"] = 12

        base_path = Path("/mnt/nvme3/ilafkir/results/training/uea_normalized/final")
        json_files = [
            base_path / "one_fits_all/evaluation_report.json",
            base_path / "time_llm/evaluation_report.json",
            base_path / "llm_few/evaluation_report.json",
            base_path / "s2ip_tempo/evaluation_report.json",
            base_path / "dp_v1/evaluation_report.json",
            base_path / "dp_v2/evaluation_report.json",
        ]

        all_results: dict[str, dict] = {}
        for file in json_files:
            with open(file, "r") as f:
                data = json.load(f)
                all_results[data["model_name"]] = data["per_dataset_results"]

        datasets: list[str] = sorted(
            set.intersection(*[set(r.keys()) for r in all_results.values()])
        )

        labels = list(all_results.keys())
        labels_for_plot = [
            "One Fits All", "Time-LLM", "LLM-Few",
            "S$^2$IP-TEMPO", "Deep Range V1", "Deep Range V2"
        ]
        metrics = ["accuracy", "f1_score", "b_accuracy"]

        all_pvals: dict[str, dict] = {}

        for metric in metrics:
            scores_matrix = [
                [all_results[clf][dataset][metric] for clf in labels]
                for dataset in datasets
            ]
            scores = np.array(scores_matrix)

            fig, _, p_vals = plot_critical_difference(
                scores=scores,
                labels=labels_for_plot,
                return_p_values=True
            )

            all_pvals[metric] = {
                "labels": labels,
                "p_values": p_vals.tolist()
            }

            png_path = base_path / f"comp_uea_{metric}_with_dp.png"
            pdf_path = base_path / f"comp_uea_{metric}_with_dp.pdf"
            fig.savefig(pdf_path, bbox_inches="tight", facecolor="white", dpi=300)
            fig.savefig(png_path, bbox_inches="tight", facecolor="white", dpi=300)

        with open(base_path / "p_values_with_dp.json", "w") as f:
            json.dump(all_pvals, f, indent=4)

    def cleanup_objects(self, objs: list) -> None:
        """Moves objects to CPU, deletes them, and frees GPU memory.

        For each object in ``objs`` the method attempts to:

        1. Call ``obj.to("cpu")`` if the object exposes a ``to`` method
           (e.g. PyTorch modules or tensors).
        2. Call ``obj.llm.to("cpu")`` if the object has a nested ``llm``
           attribute with its own ``to`` method.
        3. Delete the object reference.

        After all objects have been processed, Python's garbage collector is
        invoked, CUDA memory caches are emptied, and (if available)
        inter-process CUDA memory is collected.

        Any exception raised during cleanup of an individual object is silently
        suppressed to avoid masking upstream errors.

        Args:
            objs (list): Arbitrary list of Python objects to clean up.
                Typically contains PyTorch models, trainers, evaluators, and
                data loaders.
        """
        for o in objs:
            try:
                if hasattr(o, "to"):
                    o.to("cpu")
                if hasattr(o, "llm") and hasattr(o.llm, "to"):
                    o.llm.to("cpu")
                del o
            except Exception:
                pass

        gc.collect()
        torch.cuda.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.ipc_collect()

    
