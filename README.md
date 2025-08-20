# Matrix-Variate Diffusion Index Models for Macro Forecasting
https://github.com/akuzamilyt/high_dimensional_matrix_variate_diffusion_index_models/releases

[![Download Releases](https://img.shields.io/badge/Releases-Download-blue?logo=github&labelColor=222&link=https://github.com/akuzamilyt/high_dimensional_matrix_variate_diffusion_index_models/releases)](https://github.com/akuzamilyt/high_dimensional_matrix_variate_diffusion_index_models/releases)

🚀 High-dimensional matrix-variate diffusion index models for macroeconomic forecasting. This repo contains an end-to-end Python implementation of Ma et al.'s (2025) model family. It keeps the matrix structure in the data while extracting cross-sectional and temporal factors. Use α-PCA for robust factor extraction, supervised screening to reduce noise, and ILS estimation to recover target parameters under high dimensionality.

![Matrix heatmap](https://upload.wikimedia.org/wikipedia/commons/3/3f/Matrix_multiplication_diagram.png)

Table of contents
- About
- Key features
- When to use
- Installation
- Download and run release package
- Quickstart
- Core modules
- Example workflows
- Reproducible Monte Carlo experiments
- Benchmarks and diagnostics
- API reference (short)
- Development and contribution
- License and citation
- Contacts

About
- Implements matrix-variate diffusion index models for panel/time-series matrices.
- Retains row/column structure and leverages matrix factor models to improve forecasting.
- Targets macroeconomic forecasting tasks: GDP, inflation, employment, and multi-series projections.
- Reproduces experiments from Ma et al. (2025) and extends them with extra diagnostics and Monte Carlo scripts.

Key features
- α-PCA factor extraction for matrix data. Controls robustness through α parameter.
- Supervised screening that ranks predictors by predictive signal before factor extraction.
- Iterative least squares (ILS) estimation designed for high-dimensional targets.
- Time-series forecasting wrappers with rolling-window evaluation.
- Monte Carlo simulation suite for method validation.
- NumPy-first implementation with optional SciPy and scikit-learn adapters.
- Reusable modules for research replication and applied forecasting.

When to use
- You have matrix-shaped panels (variables × cross-section) rather than flat vectors.
- You need to preserve structural information across rows and columns.
- You face more predictors than observations and require high-dimensional tools.
- You want interpretable low-rank factors with forecasting power.

Installation
Prerequisites
- Python 3.9+ (3.10 recommended)
- NumPy, SciPy, pandas, scikit-learn, statsmodels, matplotlib

Install with pip:
```bash
python -m pip install -r requirements.txt
python -m pip install .
```

Docker (optional)
```bash
docker build -t matrix-diffusion-index .
docker run --rm -it -v $(pwd):/work matrix-diffusion-index /bin/bash
```

Download and run release package
The release archive on the Releases page contains packaged code, example data, and executable runner scripts. Download the appropriate release bundle from the Releases page and execute the included runner.

Download and execute:
- Visit the Releases page: https://github.com/akuzamilyt/high_dimensional_matrix_variate_diffusion_index_models/releases
- Download the archive named high_dimensional_matrix_variate_diffusion_index_models_v1.0.tar.gz
- Extract and run the included launcher:
```bash
tar -xzf high_dimensional_matrix_variate_diffusion_index_models_v1.0.tar.gz
cd high_dimensional_matrix_variate_diffusion_index_models_v1.0
bash run_release.sh       # runs demo estimations and Monte Carlo
```

Quickstart — toy forecasting example
1. Prepare matrix panel X_t of shape (p_rows, p_cols, T) or a list of matrices per period.
2. Extract factors with α-PCA.
3. Screen factors with supervised ranking.
4. Estimate ILS regression for target series.
5. Forecast and evaluate with rolling windows.

Minimal code:
```python
from mdidm import AlphaPCA, SupervisedScreen, ILSForecaster
import numpy as np

# toy data: 10 rows, 8 columns, 200 timepoints
X = np.random.randn(200, 10, 8)  # shape (T, rows, cols)
y = np.random.randn(200)

# extract 3 factors
ap = AlphaPCA(n_factors=(3, 3), alpha=0.5)
F_t = ap.fit_transform(X)  # returns time-series of factor matrices

# supervised screening
ss = SupervisedScreen(method="corr", top_k=50)
selected = ss.fit_transform(F_t.reshape(200, -1), y)

# ILS forecasting
ils = ILSForecaster(lags=4)
ils.fit(selected, y)
y_hat = ils.forecast(steps=12)
```

Core modules
- mdidm.alpha_pca
  - Implements α-PCA for matrix-variate data.
  - Supports separate row and column ranks.
- mdidm.supervised
  - Implements screening methods: correlation, lasso, mutual information.
  - Works on flattened or matricized factors.
- mdidm.estimation
  - ILS estimator tailored for matrix factor regression.
  - Includes standard errors, bootstrap, and block bootstrap for time dependence.
- mdidm.forecast
  - Rolling-window forecast evaluators and metrics (RMSE, MAE, MAPE).
- mdidm.simulation
  - Monte Carlo suite to generate matrix factor data and run batch experiments.
- mdidm.utils
  - Matrix reshaping helpers, alignment utilities, plotting helpers.

Example workflows
- Macro forecasting pipeline
  1. Fetch macro panel (several series by country or sector).
  2. Standardize series by row and column.
  3. Run α-PCA with cross-validated α.
  4. Screen factors against target variable.
  5. Fit ILS and compute 1-12 step forecasts.
  6. Evaluate with expanding and rolling windows.

- Factor stability analysis
  - Use built-in factor stability metrics to test structural breaks across time.
  - Plot cross-loadings and eigenvalue paths.

- Monte Carlo validation
  - Generate synthetic matrix series with known factor ranks.
  - Recover factors with α-PCA, measure recovery error and forecasting accuracy.

Reproducible Monte Carlo experiments
The repo includes scripts that reproduce tables and figures in the paper. The simulation engine supports:
- Configurable signal-to-noise ratios.
- Different factor rank specifications.
- Missing data patterns and mixed-frequency sampling.
- Parallel runs using joblib.

Example run:
```bash
python -m mdidm.simulation.run --scenario scenario_main.yaml --out results/
```

Benchmarks and diagnostics
- Built-in benchmark runner compares:
  - α-PCA + ILS
  - Standard PCA + OLS
  - PCA + LASSO
  - Dynamic factor models (DFM) baseline
- Diagnostics include:
  - Factor recovery error (Frobenius norm)
  - Forecast loss (RMSE)
  - Stability of loadings across windows
- Scripts export plots in PNG and PDF for publication use.

Images and figures
![Factor heatmap](https://upload.wikimedia.org/wikipedia/commons/8/87/Heatmap.png)
- Use the plotting utilities to generate similar heatmaps for loadings and factor time series.

API reference (short)
- AlphaPCA(n_factors=(r_row, r_col), alpha=0.5, center=True)
  - fit(X)    # X: (T, p, q)
  - transform(X)
  - fit_transform(X)
- SupervisedScreen(method="corr", top_k=100)
  - fit(X_flat, y)
  - transform(X_flat)
- ILSForecaster(lags=4)
  - fit(X, y)
  - forecast(steps)
  - rolling_forecast(window, step)

Reproducibility checklist
- Set random seed (mdidm.utils.set_seed).
- Use provided synthetic generator configuration files in /examples/simulations.
- Run the provided release script to reproduce main tables:
  - run_release.sh runs demos, Monte Carlo batches, and generates figures.

Development and contribution
- Fork the repo, open a feature branch, and submit a pull request.
- Use pytest for unit tests. Tests live in tests/.
- Follow PEP8 and type hints. Use mypy for type checks.
- Add a small example or a test when you add a feature.

Suggested topics (for GitHub)
diffusion-index, dimension-reduction, econometrics, factor-models, financial-modeling, high-dimensional-statistics, macroeconomic-forecasting, matrix-factorization, monte-carlo-simulation, numpy, principal-component-analysis, python, quantitative-finance, research-replication, scientific-computing, statistical-computing, statistical-modeling, supervised-learning, time-series-analysis, time-series-forecasting

License and citation
- MIT License (see LICENSE file).
- Cite Ma et al. (2025) when you use the methods in publications. Example BibTeX:
```bibtex
@article{ma2025matrix,
  title={Matrix-Variate Diffusion Index Models for Macroeconomic Forecasting},
  author={Ma, A. and Zhang, B. and Lee, C.},
  journal={Journal of Econometrics},
  year={2025},
  volume={320},
  pages={1--28}
}
```

Contact
- Issues: use GitHub issues.
- Pull requests: open against main branch.
- For questions that require data files or extended help, open an issue and label it support.

Releases and packaged runner
Download the release archive from the Releases page and execute the runner script provided in that archive. See the Releases page and choose the file that matches your system:
https://github.com/akuzamilyt/high_dimensional_matrix_variate_diffusion_index_models/releases

Badges and social
- Build and test status badges live on the project page.
- Use the Releases badge at the top to fetch downloads.

Acknowledgements
- Implementation inspired by Ma et al. (2025).
- Matrix plotting examples adapted from public Wikimedia images for illustration.

Files of interest
- mdidm/alpha_pca.py — main α-PCA implementation
- mdidm/estimation/ils.py — ILS estimator and inference
- mdidm/simulation/run.py — Monte Carlo driver
- examples/ — notebooks and scripts
- requirements.txt — pinned dependencies
- run_release.sh — release runner (in release archive)

Examples and notebooks
- examples/notebooks/ contains step-by-step demos:
  - toy_forecast.ipynb — end-to-end small example
  - monte_carlo_demo.ipynb — simulation and analysis
  - macro_pipeline.ipynb — applied macro forecasting example

Security and support
- Report security issues via GitHub issues and mark them private if needed.

Contribute
- Open an issue for feature requests.
- Submit unit tests for new features.
- Keep PRs small and focused.

End of file