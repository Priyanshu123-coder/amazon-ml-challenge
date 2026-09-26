# Amazon ML Challenge 2026 - Business Entity Resolution

## Project Overview
This repository contains the code and methodology for the Amazon ML Challenge 2026. The pipeline is designed to optimize for macro-averaged F0.5 while maintaining a highly efficient candidate generation phase.

## Repository Structure
* `data/`: Contains the raw and processed datasets (ignored by git).
* `notebooks/`: Jupyter notebooks for EDA and quick experiments.
* `src/`: Core Python modules for the pipeline.
  * `blocking.py`: Two-stage candidate generation (Deterministic + TF-IDF).
  * `features.py`: Feature engineering (Levenshtein, Jaro-Winkler, Jaccard).
  * `model.py`: Matching model training and F0.5 threshold optimization.
* `decision_log.md`: Ongoing log of validation results and architectural choices.
* `requirements.txt`: Python dependencies for reproducibility.
