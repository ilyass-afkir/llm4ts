.. Ilyass Afkir documentation master file, created by
   sphinx-quickstart on Thu Apr  2 14:35:00 2026.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Master Thesis Code Documentation
================================

**Large Language Models (LLM) for Range Prediction of Electric Trucks in Pre-Development Software and Electronics**

Accurate range prediction for electric trucks is crucial for energy management, operational planning, and driver confidence in commercial electromobility. As electric truck fleets grow in size and variant diversity, the scalability of existing bespoke machine learning models becomes a critical limitation, as each model must be individually maintained per vehicle variant.

Motivated by the rapid progress of foundation models in natural language processing and computer vision, this thesis investigates whether Large Language Models can be adapted to multivariate time series data to provide a scalable and unified classification architecture for electric truck applications.

To this end, this thesis introduces **DeepRange**, a novel LLM-based architecture for multivariate time series classification, evaluated on real-world electric truck scenario classification across three domains: highway type, ambient temperature, and weather condition. Two candidate architectures with distinct data representation strategies are first compared on neutral multivariate time series classification benchmarks before being applied to the electric truck domain, ensuring model selection remains free from domain-specific assumptions.

DeepRange is subsequently proposed in two backbone configurations (1B and 8B parameters) and evaluated against the baselines **InceptionTime** and **MiniRocket** across 12 experimental settings, covering two sequence lengths (100s and 500s) and two data regimes (full training set and 10% reduced set).

Results demonstrate that DeepRange is a viable and scalable alternative to bespoke per-variant ML models for scenario classification. DeepRange 8B outperforms baselines under sufficient data, while DeepRange 1B provides a favorable trade-off between performance and computational cost. The findings provide a foundation for integrating LLM-based multivariate time series classification into electric truck range prediction pipelines.

.. toctree::
   :maxdepth: 1
   :hidden:

   modules
