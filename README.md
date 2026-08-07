# Binary Logistic Regression from Scratch --- Pure NumPy

A from-scratch implementation of binary Logistic Regression using NumPy
for the mathematical and optimization core.

The project makes the mechanics of logistic regression explicit rather
than relying on a high-level estimator such as
`sklearn.linear_model.LogisticRegression`.

The implementation covers:

-   sigmoid activation;
-   binary cross-entropy loss;
-   L2 (Ridge) regularization;
-   analytical gradient computation;
-   gradient descent;
-   vectorized matrix operations with NumPy;
-   feature standardization before optimization;
-   comparison of learned coefficients with and without regularization.

------------------------------------------------------------------------

## Mathematical Formulation

### 1. Sigmoid Activation

The sigmoid function maps the linear model output to a probability:

$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

For a feature matrix $X$, weight vector $W$, and bias $b$:

$$
A = \sigma(XW + b)
$$

where $A$ contains the predicted probabilities for the positive class.

### 2. Binary Cross-Entropy with L2 Regularization

The objective function combines binary cross-entropy with an L2 penalty:

$$
J(W,b) =
-\frac{1}{n}
\sum_{i=1}^{n}
\left[
y_i \log(A_i)
+
(1-y_i)\log(1-A_i)
\right]
+
\frac{\alpha}{2n}
\sum_{j=1}^{m}w_j^2
$$

The bias term is deliberately excluded from regularization.

### 3. Analytical Gradients

Applying the chain rule gives the gradients used for parameter updates:

$$
\frac{\partial J}{\partial W}
=
\frac{1}{n}X^T(A-Y)
+
\frac{\alpha}{n}W
$$

and

$$
\frac{\partial J}{\partial b}
=
\frac{1}{n}
\sum_{i=1}^{n}(A_i-Y_i)
$$

These gradients are implemented directly with NumPy matrix operations.

------------------------------------------------------------------------

## Implementation

The core model is implemented from scratch using NumPy.

Key implementation details:

-   numerically protected sigmoid computation using `np.clip`;
-   clipping of predicted probabilities before taking logarithms;
-   vectorized forward propagation;
-   manually derived analytical gradients;
-   binary cross-entropy loss;
-   optional L2 regularization;
-   gradient-descent parameter updates;
-   feature standardization before optimization.

The model itself does not use a high-level Logistic Regression
estimator.

------------------------------------------------------------------------

## Dataset and Preprocessing

A synthetic binary classification dataset was generated with:

-   1,000 samples;
-   10 features;
-   fixed random seed `42`.

``` python
X_raw, y_raw = make_classification(
    n_samples=1000,
    n_features=10,
    random_state=42
)
```

The features were standardized before gradient descent:

``` python
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)
```

Feature scaling is important here because gradient descent is sensitive
to differences in feature magnitude.

------------------------------------------------------------------------

## Experiments

### Experiment 1 --- No Regularization

The first experiment uses:

``` text
alpha = 0.0
```

This provides a baseline where the parameters are optimized only
according to the data loss.

Observed parameters:

``` text
Weights:
[-0.490, 0.199, -1.159, -0.007, -0.044,
 -0.168, 1.628, 0.011, -1.071, 0.088]

Bias:
0.163
```

Observed accuracy on the same dataset used for optimization:

``` text
85.9%
```

------------------------------------------------------------------------

### Experiment 2 --- L2 Regularization

The second experiment uses a strong L2 penalty:

``` text
alpha = 50.0
```

The penalty constrains coefficient magnitudes during optimization.

Observed parameters:

``` text
Weights:
[-0.386, 0.105, -0.749, -0.004, -0.008,
 -0.067, 0.990, -0.002, -0.511, 0.032]

Bias:
0.070
```

Observed accuracy on the same dataset:

``` text
87.0%
```

------------------------------------------------------------------------

## Weight Compression Analysis

The project explicitly compares the learned coefficients before and
after L2 regularization.

    Feature   Without L2   With L2   Absolute Difference
  --------- ------------ --------- ---------------------
          1       -0.490    -0.386                 0.104
          2        0.199     0.105                 0.093
          3       -1.159    -0.749                 0.410
          4       -0.007    -0.004                 0.003
          5       -0.044    -0.008                 0.036
          6       -0.168    -0.067                 0.101
          7        1.628     0.990                 0.638
          8        0.011    -0.002                 0.013
          9       -1.071    -0.511                 0.560
         10        0.088     0.032                 0.056

The largest changes occur for the strongest coefficients. For example,
the coefficient of Feature 7 decreases from `1.628` to `0.990`.

This makes the effect of L2 regularization visible directly in the
learned parameters.

------------------------------------------------------------------------

## Results

  Model                              Accuracy
  -------------------------------- ----------
  No regularization                     85.9%
  L2 regularization (`alpha=50`)        87.0%

The experiment demonstrates that changing the regularization strength
substantially changes the learned coefficient vector and, in this run,
also changes the observed classification accuracy.

> **Important:** both accuracy values are measured on the same dataset
> used to optimize the model. They should therefore be interpreted as
> training accuracy, not as an estimate of out-of-sample generalization
> performance.

------------------------------------------------------------------------

## Technical Stack

-   **Python 3**
-   **NumPy** --- model implementation and vectorized numerical
    computation
-   **Pandas** --- coefficient comparison and result analysis
-   **scikit-learn** --- synthetic dataset generation, feature
    standardization, and accuracy metrics

------------------------------------------------------------------------

## Repository Structure

``` text
numpy-logistic-regression-from-scratch/
│
├── README.md
└── lr_log_regression.ipynb
```

------------------------------------------------------------------------

## What This Project Demonstrates

-   Logistic regression mathematics
-   Binary cross-entropy
-   Gradient descent
-   Analytical gradient derivation
-   Matrix and vector operations with NumPy
-   L2 regularization
-   Numerical stability considerations
-   Feature scaling for gradient-based optimization
-   Interpretation of model coefficients
-   Experimental analysis of regularization effects
