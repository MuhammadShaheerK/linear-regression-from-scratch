# Linear Regression From Scratch

A hands-on implementation of **Linear Regression from scratch using NumPy and Gradient Descent**, followed by a comparison with Scikit-learn's `LinearRegression`.

The goal of this project is to understand what happens inside a linear regression model instead of treating the implementation as a black box.

---

## Overview

This project uses the **California Housing dataset** to build and evaluate a linear regression model.

The project covers:

- Data inspection and cleaning
- Exploratory Data Analysis (EDA)
- Train/test splitting
- Feature standardization
- Linear Regression from scratch
- Batch Gradient Descent
- Cost function and gradient calculation
- Model evaluation
- Prediction visualization
- Comparison with Scikit-learn

---

## Dataset

The project uses the **California Housing dataset**.

### Target

`median_house_value`

### Numerical Features

- `longitude`
- `latitude`
- `housing_median_age`
- `total_rooms`
- `total_bedrooms`
- `population`
- `households`
- `median_income`

The categorical feature `ocean_proximity` was removed for this first implementation so that the model could operate directly on numerical features.

---

## Data Preparation

The dataset was inspected using Pandas through:

- `head()`
- `describe()`
- `info()`
- Missing-value checks
- Duplicate checks

There were missing values in `total_bedrooms`. These rows were removed before training.

Exploratory analysis was performed using histograms, box plots, and feature-vs-target scatter plots.

The dataset was then split into:

- **80% training data**
- **20% testing data**

Feature standardization was performed using the **training-set mean and standard deviation**:

$$
x_{\text{scaled}} =
\frac{x-\mu_{\text{train}}}
{\sigma_{\text{train}}}
$$

The same training statistics were applied to the test set to avoid data leakage.

### Median Income Distribution

![Median Income Distribution](images/median_income_distribution.png)

---

# Linear Regression From Scratch

The prediction function is:

$$
\hat{y}=XW+b
$$

where:

- $X$ = feature matrix
- $W$ = weight vector
- $b$ = bias
- $\hat{y}$ = predicted values

The weights and bias were initialized and optimized using **Batch Gradient Descent**.

---

## Cost Function

The model uses the Mean Squared Error-based cost function:

$$
J(W,b)=
\frac{1}{2m}
\sum_{i=1}^{m}
(\hat{y}_i-y_i)^2
$$

The factor $\frac{1}{2}$ simplifies the derivative of the squared error.

---

## Gradient Descent

The weight gradient is:

$$
\nabla_W J =
\frac{1}{m}X^T(\hat{y}-y)
$$

The bias gradient is:

$$
\nabla_b J =
\frac{1}{m}\sum_{i=1}^{m}(\hat{y}_i-y_i)
$$

The parameters are updated using:

$$
W \leftarrow W-\alpha\nabla_WJ
$$

$$
b \leftarrow b-\alpha\nabla_bJ
$$

where $\alpha$ is the learning rate.

For the current experiment:

- Learning rate: `0.01`
- Iterations: `1999`

---

## Gradient Descent Convergence

The cost was recorded during training and plotted against the number of iterations.

![Cost Function](images/cost_function.png)

The decreasing cost demonstrates that Gradient Descent is progressively reducing the training loss and approaching a stable solution.

---

# Model Evaluation

The trained model was evaluated on the unseen test set using:

### Mean Squared Error

$$
MSE=
\frac{1}{n}
\sum_{i=1}^{n}
(y_i-\hat{y}_i)^2
$$

MSE penalizes larger errors more heavily because the errors are squared.

### Root Mean Squared Error

$$
RMSE=\sqrt{MSE}
$$

RMSE converts the error back to the original target units, making it easier to interpret.

### R² Score

$$
R^2=
1-
\frac{
\sum(y_i-\hat{y}_i)^2
}{
\sum(y_i-\bar{y})^2
}
$$

R² measures how much of the variation in the target is explained by the model relative to predicting the mean target value.

---

# Results

The initial Gradient Descent implementation was evaluated against Scikit-learn's Linear Regression implementation.

| Model | RMSE | R² |
|---|---:|---:|
| From Scratch | ~$70,765 | ~0.634 |
| Scikit-learn | ~$70,156 | ~0.640 |

The Scikit-learn implementation performs slightly better in the current experiment.

However, increasing the number of Gradient Descent iterations progressively brought the custom implementation closer to the Scikit-learn solution.

This demonstrates an important optimization concept:

> Gradient Descent can approach the optimal linear regression parameters when given an appropriate learning rate and enough iterations.

---

## From Scratch vs Scikit-learn

### From-Scratch Model

![From Scratch Predictions](images/predictions_from_scratch.png)

### Scikit-learn Model

![Scikit-learn Predictions](images/predictions_sklearn.png)

Both models capture the general relationship between the input features and house prices, although substantial prediction error remains.

---

# Technologies

- Python
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Jupyter Notebook

---

# Project Structure

```text
linear-regression/
│
├── data/
│   └── california_housing.csv
│
├── images/
│   ├── median_income_distribution.png
│   ├── cost_function.png
│   ├── predictions_from_scratch.png
│   └── predictions_sklearn.png
│
├── linear_regression.ipynb
├── README.md
└── requirements.txt