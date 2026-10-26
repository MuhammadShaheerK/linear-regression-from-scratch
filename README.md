# Linear Regression from Scratch

Batch gradient descent implemented with NumPy on the bundled California housing dataset, compared with sklearn LinearRegression.

## Reproduced results

80/20 split, seed 42; 20,433 complete records, with 16,346 training and 4,087 test records.

| Model | Updates | MSE | RMSE | R² |
|---|---:|---:|---:|---:|
| Scratch | 500 | 5,647,828,380.13 | 75,152.04 | 0.587002 |
| Scratch | 2000 | 5,007,515,706.94 | 70,763.80 | 0.633825 |
| Scratch | 5000 | 4,922,703,474.58 | 70,161.98 | 0.640026 |
| sklearn | — | 4,921,881,237.63 | 70,156.12 | 0.640087 |

Full precision values are in [results.json](results.json). These replace the original results: the update counts are now exact and R² is calculated using residual sum of squares rather than residual variance.

## Run

Use Python 3.12, from the repository root:

~~~bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux, instead:
# source .venv/bin/activate
python -m pip install -r requirements.txt
python experiment.py --plots
python -m pytest -q
~~~

The script runs all three experiments from zero initialization, writes results.json, and refreshes images/ when --plots is supplied. Dataset paths are relative to the script, not your current working directory. Use --output another-file.json to change the results destination.

Open notebook.ipynb in VS Code or another Jupyter-compatible editor with this environment, restart the kernel, and Run All from the repository root. Install a standalone notebook UI separately if needed: python -m pip install jupyterlab.

## Method

1. Read data/california_housing.csv; drop rows containing missing values.
2. Omit ocean_proximity for a numerical-only baseline.
3. Split using test_size=0.2 and random_state=42.
4. Fit StandardScaler on training features only.
5. Learn weights and an intercept with batch gradient descent at learning rate 0.01.
6. Perform exactly 500, 2000, and 5000 updates in independent runs.
7. Evaluate every model on the same test set using MSE, RMSE, and sklearn.metrics.r2_score.

Training cost is MSE / 2. R² is 1 - SSE / SST; a constant prediction bias must contribute to SSE.

## Figures

![Convergence at 5000 updates](images/cost_function_5000.png)
![Scratch predictions](images/predictions_from_scratch_5000.png)
![sklearn predictions](images/predictions_sklearn.png)

## Files and tests

- model.py: gradient descent implementation.
- experiment.py: reproducible experiments and figure generation.
- notebook.ipynb: educational walkthrough.
- data/: bundled dataset; no network download required.
- images/: figures regenerated from the corrected code.
- tests/: checks for biased-prediction R², exact iteration counts, intercept learning, and convergence toward sklearn.

## Limitations

This is a learning baseline, not a production housing valuation model. It omits categorical features, drops missing rows, and uses a random split rather than a geographic or temporal evaluation. The experiments illustrate optimization convergence; future tuning needs a validation set rather than selecting settings on test performance.

Dataset: California housing data described by [Hands-On Machine Learning](https://github.com/ageron/handson-ml2/tree/master/datasets/housing). The source's terms and attribution apply to the bundled data.

## Automated checks

Run python verify_notebook.py to execute every notebook cell in a fresh kernel. GitHub Actions runs tests and this notebook check on pushes and pull requests.
