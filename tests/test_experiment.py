import numpy as np
from experiment import evaluate, run_experiment
from model import LinearRegressionScratch


def test_r2_accounts_for_constant_prediction_bias():
    # A variance-only error formula incorrectly gives 1.0 here.
    result = evaluate(np.array([1., 2., 3.]), np.array([2., 3., 4.]))
    assert result["r2"] == -0.5


def test_exact_update_count_and_intercept():
    model = LinearRegressionScratch(learning_rate=0.1, iterations=200).fit(np.zeros((5, 1)), np.full(5, 7.))
    assert len(model.cost_history_) == 200
    np.testing.assert_allclose(model.predict([[0]]), [7], atol=1e-6)


def test_full_experiment_converges_toward_reference():
    results, (models, _, _) = run_experiment()
    scores = results["models"]
    assert scores["5000"]["rmse"] < scores["500"]["rmse"]
    assert abs(scores["5000"]["r2"] - scores["sklearn"]["r2"]) < 0.001
    assert abs(scores["5000"]["rmse"] - scores["sklearn"]["rmse"]) < 20
    assert len(models["2000"].cost_history_) == 2000
