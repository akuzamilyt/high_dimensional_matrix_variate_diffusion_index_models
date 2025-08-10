# README.md

# High-Dimensional Matrix-Variate Diffusion Index Models

<!-- PROJECT SHIELDS -->
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)
[![Type Checking: mypy](https://img.shields.io/badge/type_checking-mypy-blue)](http://mypy-lang.org/)
[![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=flat&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=flat&logo=numpy&logoColor=white)](https://numpy.org/)
[![SciPy](https://img.shields.io/badge/SciPy-%23025596?style=flat&logo=scipy&logoColor=white)](https://scipy.org/)
[![Statsmodels](https://img.shields.io/badge/Statsmodels-150458.svg?style=flat&logo=python&logoColor=white)](https://www.statsmodels.org/stable/index.html)
[![Scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-%23F37626.svg?style=flat&logo=Jupyter&logoColor=white)](https://jupyter.org/)
[![arXiv](https://img.shields.io/badge/arXiv-2508.04259-b31b1b.svg)](https://arxiv.org/abs/2508.04259)
[![Research](https://img.shields.io/badge/Research-Macroeconomic%20Forecasting-green)](https://github.com/chirindaopensource/high_dimensional_matrix_variate_diffusion_index_models)
[![Discipline](https://img.shields.io/badge/Discipline-Econometrics-blue)](https://github.com/chirindaopensource/high_dimensional_matrix_variate_diffusion_index_models)
[![Methodology](https://img.shields.io/badge/Methodology-Factor%20Models-orange)](https://github.com/chirindaopensource/high_dimensional_matrix_variate_diffusion_index_models)
[![Year](https://img.shields.io/badge/Year-2025-purple)](https://github.com/chirindaopensource/high_dimensional_matrix_variate_diffusion_index_models)

**Repository:** `https://github.com/chirindaopensource/high_dimensional_matrix_variate_diffusion_index_models`

**Owner:** 2025 Craig Chirinda (Open Source Projects)

This repository contains an **independent**, professional-grade Python implementation of the research methodology from the 2025 paper entitled **"High-Dimensional Matrix-Variate Diffusion Index Models for Time Series Forecasting"** by:

*   Zhiren Ma
*   Qian Zhao
*   Riquan Zhang
*   Zhaoxing Gao

The project provides a complete, end-to-end computational framework for forecasting a scalar time series using a high-dimensional, matrix-valued panel of predictors. It moves beyond traditional vectorized factor models by preserving the intrinsic row-column structure of the data, offering a more powerful and nuanced approach to dimension reduction in modern data-rich environments. The goal is to provide a transparent, robust, and computationally efficient toolkit for researchers and practitioners to replicate, validate, and extend the paper's findings.

## Table of Contents

- [Introduction](#introduction)
- [Theoretical Background](#theoretical-background)
- [Features](#features)
- [Methodology Implemented](#methodology-implemented)
- [Core Components (Notebook Structure)](#core-components-notebook-structure)
- [Key Callable: run_complete_study](#key-callable-run_complete_study)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Input Data Structure](#input-data-structure)
- [Usage](#usage)
- [Output Structure](#output-structure)
- [Project Structure](#project-structure)
- [Customization](#customization)
- [Contributing](#contributing)
- [License](#license)
- [Citation](#citation)
- [Acknowledgments](#acknowledgments)

## Introduction

This project provides a Python implementation of the methodologies presented in the 2025 paper "High-Dimensional Matrix-Variate Diffusion Index Models for Time Series Forecasting." The core of this repository is the iPython Notebook `high_dimensional_matrix_variate_diffusion_index_models_draft.ipynb`, which contains a comprehensive suite of functions to replicate the paper's findings, from initial data validation and cleansing to the final generation of performance tables, diagnostic plots, and a full reproducibility report.

Traditional diffusion index models require vectorizing predictor panels, which can destroy valuable structural information (e.g., the relationship between different economic indicators for a single country). This project implements a matrix-variate approach that preserves this structure, potentially leading to more powerful factors and more accurate forecasts.

This codebase enables users to:
-   Rigorously validate and prepare complex panel datasets, ensuring stationarity and preventing data leakage in a forecasting context.
-   Extract low-dimensional latent factor matrices from a high-dimensional data tensor using the flexible α-PCA methodology.
-   Estimate a bilinear forecasting model using a numerically stable Iterative Least Squares (ILS) algorithm.
-   Apply a novel supervised screening technique to refine the predictor set and improve forecast accuracy.
-   Conduct a full-scale Monte Carlo simulation to validate the statistical properties of the estimators.
-   Perform a comprehensive empirical study, including benchmark comparisons and robustness checks.
-   Automatically generate all key tables and figures from the paper for direct comparison and validation.

## Theoretical Background

The implemented methods are grounded in modern high-dimensional econometrics, extending classical factor model theory to matrix- and tensor-valued time series.

**1. The Matrix-Variate Diffusion Index Model:**
The core model assumes that a high-dimensional predictor matrix $X_t \in \mathbb{R}^{p \times q}$ and a future scalar outcome $y_{t+h}$ are driven by a common low-dimensional latent factor matrix $F_t \in \mathbb{R}^{k \times r}$.
-   **Observation Equation:** $X_t = R F_t C' + E_t$
-   **Forecasting Equation:** $y_{t+h} = \alpha' F_t \beta + e_{t+h}$

**2. α-Principal Component Analysis (α-PCA):**
Unlike standard PCA which only considers the covariance matrix (second moments), α-PCA constructs aggregation matrices that are a weighted average of both first and second moments of the data. This allows the factor extraction to be sensitive to both the mean structure and the variance structure of the predictors. The key constructs are the moment aggregation matrices:
$$
\widehat{\boldsymbol{M}}_R = \frac{1}{pq} \left[ (1+\alpha) \overline{\boldsymbol{X}} \overline{\boldsymbol{X}}^\prime + \frac{1}{T} \sum_{t=1}^T (\boldsymbol{X}_t - \overline{\boldsymbol{X}}) (\boldsymbol{X}_t - \overline{\boldsymbol{X}})^\prime \right]
$$
The loading matrices $R$ and $C$ are derived from the eigendecomposition of $\widehat{\boldsymbol{M}}_R$ and its column-wise equivalent $\widehat{\boldsymbol{M}}_C$.

**3. Supervised Screening:**
To improve the signal-to-noise ratio, the paper proposes a supervised pre-processing step. It computes the correlation of each individual predictor series $x_{ij,t}$ with the target $y_t$. Rows (e.g., countries) and columns (e.g., indicators) with low average absolute correlation are removed before the α-PCA is applied, focusing the dimension reduction on the most relevant parts of the data.

## Features

The provided iPython Notebook (`high_dimensional_matrix_variate_diffusion_index_models_draft.ipynb`) implements the full research pipeline, including:

-   **Data Pipeline:** A robust, leak-free validation and preparation module that performs stationarity testing, transformation, and centralization appropriate for forecasting.
-   **High-Performance Analytics:** Elite-grade, vectorized implementations of the α-PCA and ILS algorithms using advanced NumPy and SciPy features.
-   **Statistical Rigor:** A complete suite of benchmark models (AR, Vec-OLS, Vec-Lasso) and a robust Diebold-Mariano test with small-sample corrections for fair and accurate model comparison.
-   **Automated Orchestration:** A master function that runs the entire end-to-end workflow, including the empirical study, simulations, and robustness checks, with a single call.
-   **Comprehensive Reporting:** Automated generation of publication-quality summary tables (replicating Tables 1, 2, 4, 5 from the paper), diagnostic plots, and a full reproducibility report.
-   **Full Research Lifecycle:** The codebase covers the entire research process from data ingestion to final output generation, providing a complete and transparent replication package.

## Methodology Implemented

The core analytical steps directly implement the methodology from the paper:

1.  **Data Validation and Preparation (Tasks 1-2):** The pipeline ingests the raw panel data, performs structural and quality checks, and applies a leak-free stationarity and centralization protocol.
2.  **α-PCA Factor Extraction (Tasks 3-6):** It computes sample statistics, constructs the moment aggregation matrices, performs eigendecomposition to find the loadings, and projects the data to recover the latent factor matrices.
3.  **LSE Parameter Estimation (Tasks 7-8):** It prepares the training data and uses a numerically stable Iterative Least Squares algorithm to estimate the forecasting parameters $\alpha$ and $\beta$.
4.  **Supervised Screening (Tasks 9-10):** It computes training-data-only correlations, applies thresholds to filter the data, and prepares a refined data tensor.
5.  **Out-of-Sample Forecasting and Evaluation (Tasks 12-14):** It generates forecasts on unseen data and evaluates them using MSFE and the Diebold-Mariano test.
6.  **Simulation Study (Tasks 16-17):** It implements the full Monte Carlo simulation framework to generate synthetic data and validate the estimators' statistical properties.
7.  **Orchestration and Reporting (Tasks 18-20):** Master functions orchestrate the empirical study, robustness checks, and the generation of all final tables and reports.

## Core Components (Notebook Structure)

The `high_dimensional_matrix_variate_diffusion_index_models_draft.ipynb` notebook is structured as a logical pipeline with modular functions for each task, from Task 1 (Data Validation) to Task 20 (Results Compilation).

## Key Callable: run_complete_study

The central function in this project is `run_complete_study`. It orchestrates the entire analytical workflow from raw data to a final, comprehensive report object.

```python
def run_complete_study(
    country_data: Dict[str, pd.DataFrame],
    y_series: pd.Series,
    study_manifest: Dict[str, Any],
    run_empirical_study: bool = True,
    run_simulation_study: bool = True,
    generate_reports: bool = True
) -> Dict[str, Any]:
    """
    Executes the entire research pipeline from data to final report.
    """
    # ... (implementation is in the notebook)
```

## Prerequisites

-   Python 3.9+
-   Core dependencies: `pandas`, `numpy`, `scipy`, `statsmodels`, `scikit-learn`, `matplotlib`, `joblib`, `tqdm`.

## Installation

1.  **Clone the repository:**
    ```sh
    git clone https://github.com/chirindaopensource/high_dimensional_matrix_variate_diffusion_index_models.git
    cd high_dimensional_matrix_variate_diffusion_index_models
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install Python dependencies:**
    ```sh
    pip install pandas numpy scipy statsmodels scikit-learn matplotlib joblib tqdm
    ```

## Input Data Structure

The pipeline requires two primary data inputs passed to the `run_complete_study` function:

1.  **`country_data`**: A Python dictionary where keys are string identifiers for the row entities (e.g., countries) and values are `pandas.DataFrame`s. Each DataFrame must have a `DatetimeIndex` and identical column names representing the predictor variables.
2.  **`y_series`**: A `pandas.Series` containing the scalar target variable, with a `DatetimeIndex` that is identical to those in the `country_data` DataFrames.
3.  **`study_manifest`**: A nested Python dictionary that controls all parameters of the analysis. A fully specified example is provided in the notebook.

## Usage

The `high_dimensional_matrix_variate_diffusion_index_models_draft.ipynb` notebook provides a complete, step-by-step guide. The core workflow is:

1.  **Prepare Inputs:** Load your panel data into the required dictionary and Series formats. Define the `study_manifest` dictionary.
2.  **Execute Pipeline:** Call the master orchestrator function:
    ```python
    final_results = run_complete_study(
        country_data=my_raw_country_data,
        y_series=my_raw_y_series,
        study_manifest=my_study_manifest,
        run_empirical_study=True,
        run_simulation_study=False, # Optional: very time-consuming
        generate_reports=True
    )
    ```
3.  **Inspect Outputs:** Programmatically access any result from the returned `final_results` dictionary. For example, to view the main results table for the unscreened model:
    ```python
    table4_panel_a = final_results['reports']['Table 4']['Panel A']
    # In a Jupyter Notebook, this will render the styled table
    display(table4_panel_a)
    ```

## Output Structure

The `run_complete_study` function returns a single, comprehensive dictionary with the following top-level keys:

-   `empirical_study`: A deeply nested dictionary containing all raw numerical results from the empirical analysis, including performance DataFrames for the main model (screened and unscreened), benchmark results, and significance tests.
-   `simulation_study`: A dictionary containing the raw `pd.DataFrame` from the Monte Carlo simulation (if run).
-   `reports`: A dictionary containing all generated outputs, including styled `pd.DataFrame` objects for each table, `matplotlib` figure objects for plots, and the final reproducibility report.

## Project Structure

```
high_dimensional_matrix_variate_diffusion_index_models/
│
├── high_dimensional_matrix_variate_diffusion_index_models_draft.ipynb  # Main implementation notebook   
├── requirements.txt                                                      # Python package dependencies
├── LICENSE                                                               # MIT license file
└── README.md                                                             # This documentation file
```

## Customization

The pipeline is highly customizable via the master `study_config` dictionary. Users can easily modify:
-   The `alpha_grid` and `factor_dimensions_grid` for the empirical study.
-   The `train_test_split_config` to change the out-of-sample period.
-   All parameters for the Monte Carlo simulations (`p`, `q`, `T`, DGPs, etc.).
-   The `screening_thresholds` for the supervised refinement step.

## Contributing

Contributions are welcome. Please fork the repository, create a feature branch, and submit a pull request with a clear description of your changes. Adherence to PEP 8, type hinting, and comprehensive docstrings is required.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Citation

If you use this code or the methodology in your research, please cite the original paper:

```bibtex
@article{ma2025high,
  title={High-Dimensional Matrix-Variate Diffusion Index Models for Time Series Forecasting},
  author={Ma, Zhiren and Zhao, Qian and Zhang, Riquan and Gao, Zhaoxing},
  journal={arXiv preprint arXiv:2508.04259},
  year={2025}
}
```

For the implementation itself, you may cite this repository:
```
Chirinda, C. (2025). A Python Implementation of "High-Dimensional Matrix-Variate Diffusion Index Models for Time Series Forecasting". 
GitHub repository: https://github.com/chirindaopensource/high_dimensional_matrix_variate_diffusion_index_models
```

## Acknowledgments

-   Credit to Zhiren Ma, Qian Zhao, Riquan Zhang, and Zhaoxing Gao for their clear and insightful research.
-   Thanks to the developers of the scientific Python ecosystem (`numpy`, `pandas`, `scipy`, `statsmodels`, `scikit-learn`) that makes this work possible.

--

*This README was generated based on the structure and content of `high_dimensional_matrix_variate_diffusion_index_models_draft.ipynb` and follows best practices for research software documentation.*
