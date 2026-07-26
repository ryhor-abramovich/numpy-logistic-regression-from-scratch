# numpy-logistic-regression-from-scratch
Binary Logistic Regression classifier built from scratch using pure NumPy vector algebra, matrix calculus, and custom L2 (Ridge) regularization.

# 🧮 Custom Logistic Regression Engine from Scratch (Pure NumPy)

A production-grade implementation of a Binary Logistic Regression classifier built entirely from scratch using mathematical optimization, matrix calculus, and pure **NumPy vector algebra**. 

The goal of this project is to expose the low-level mechanics of gradient descent optimization, avoiding high-level abstractions like Scikit-Learn.

---

## ⚡ Core Engineering & Mathematical Features
* **Vectorized Forward Propagation:** Fully vectorized computation of linear combinations ($Z = XW + b$) and probability mapping via a customized, numerically clipped Sigmoid activation function.
* **Analytical Gradient Descent:** Exact computation of partial derivatives ($dW$, $db$) derived via the matrix Chain Rule.
* **Custom L2 (Ridge) Regularization:** Integrated structural weight decay into the gradient calculation to actively penalize extreme coefficient magnitudes and combat overfitting.
* **YAGNI Compliant Codebase:** Highly modular and clean execution where a single unified `fit()` architecture dynamically toggles regularized vs. non-regularized states based on the default $\alpha=0.0$ parameter.

---

## 📊 Experimental Results & Weight Compression

The engine was validated using a synthetic high-variance classification dataset (1,000 samples, 10 features). Introducing a strict L2 penalty ($\alpha = 50.0$) dynamically altered the geometric steepness of the decision boundary:

* **Baseline Model ($\alpha = 0.0$):** Global Accuracy = **85.9%**
* **Regularized Model ($\alpha = 50.0$):** Global Accuracy = **87.0%**

### Weight Decay Breakdown (`df_weights`):
Feature_ID  Vanilla_Weights  Ridge_Weights  Abs_Difference1           
1.           -0.490            -0.386           0.1042            
2.            0.199             0.105           0.0933           
3.           -1.159            -0.749           0.4107            
7.            1.628             0.990           0.6389           
9.           -1.071            -0.511           0.560

*Statistical Learning Note:* Shifting the optimization constraints forced aggressive, noisy weights (such as Feature 7 and 9) to shrink toward zero. This effectively smoothed the classifier plane and systematically **improved generalization performance on unseen evaluation tokens by 1.1%**.

---

## 📁 Repository Structure
* `numpy_logistic_regression.ipynb` — Complete standalone Jupyter Notebook containing core math blocks rendered via LaTeX, vectorized functions, data preprocessing, and comparative test runs.

## 🛠️ Technical Stack
* **Language:** Python 3
* **Computation Engine:** NumPy (Vectorized arrays, dot-product operators, analytical masking)
* **Validation & Pipeline:** Pandas, Scikit-Learn (`StandardScaler`, `make_classification`, `accuracy_score`)
  
