"""Reproducible California housing baseline; no download is needed."""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from model import LinearRegressionScratch

ROOT = Path(__file__).resolve().parent
ITERATIONS = (500, 2000, 5000)


def prepare_data():
    df = pd.read_csv(ROOT / "data" / "california_housing.csv").dropna()
    X = df.drop(columns=["median_house_value", "ocean_proximity"])
    y = df["median_house_value"]
    train, test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    return df, scaler.fit_transform(train), scaler.transform(test), y_train.to_numpy(), y_test.to_numpy()


def evaluate(y, predicted):
    mse = mean_squared_error(y, predicted)
    return {"mse": float(mse), "rmse": float(np.sqrt(mse)), "r2": float(r2_score(y, predicted))}


def run_experiment():
    df, train, test, y_train, y_test = prepare_data()
    models = {}
    results = {"dataset": {"rows_after_dropping_missing": len(df), "train_rows": len(y_train),
                           "test_rows": len(y_test), "random_state": 42}, "models": {}}
    for n in ITERATIONS:
        model = LinearRegressionScratch(iterations=n).fit(train, y_train)
        models[str(n)] = model
        results["models"][str(n)] = evaluate(y_test, model.predict(test))
    reference = LinearRegression().fit(train, y_train)
    models["sklearn"] = reference
    results["models"]["sklearn"] = evaluate(y_test, reference.predict(test))
    return results, (models, test, y_test)


def save_plots(models, test, y_test, output_dir=None):
    import matplotlib.pyplot as plt
    output_dir = Path(output_dir or ROOT / "images")
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(ROOT / "data" / "california_housing.csv")
    fig, ax = plt.subplots()
    df["median_income"].hist(bins=50, ax=ax)
    ax.set(xlabel="Median income (dataset units)", ylabel="Frequency", title="Median Income Distribution")
    fig.tight_layout()
    fig.savefig(output_dir / "median_income_distribution.png")
    plt.close(fig)
    for label, model in models.items():
        if label != "sklearn":
            fig, ax = plt.subplots()
            ax.plot(np.arange(1, len(model.cost_history_) + 1), model.cost_history_)
            ax.set(xlabel="Iteration", ylabel="Training cost (MSE / 2)", title=f"Gradient Descent: {label} Updates")
            fig.tight_layout()
            fig.savefig(output_dir / f"cost_function_{label}.png")
            plt.close(fig)
        predicted = model.predict(test)
        fig, ax = plt.subplots()
        ax.scatter(y_test, predicted, s=5, alpha=0.25)
        lower, upper = min(y_test.min(), predicted.min()), max(y_test.max(), predicted.max())
        ax.plot([lower, upper], [lower, upper], "r--")
        ax.set(xlabel="Actual house value", ylabel="Predicted house value", title=f"Predictions: {label}")
        fig.tight_layout()
        filename = "predictions_sklearn.png" if label == "sklearn" else f"predictions_from_scratch_{label}.png"
        fig.savefig(output_dir / filename)
        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "results.json")
    parser.add_argument("--plots", action="store_true")
    args = parser.parse_args()
    results, (models, test, y_test) = run_experiment()
    args.output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    if args.plots:
        save_plots(models, test, y_test)
    print(json.dumps(results, indent=2))
