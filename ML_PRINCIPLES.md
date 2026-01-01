# Machine Learning Principles

## Appendix
- [Core Algorithms]()
- [Feature Scaling](#feature-scaling)
    - [Normalization](#normalization)
    - [Standardization](#standardization)
    - [Robust Scaling](#robust-scaling)
    - [Log Scaling](#log-scaling)
- [Loss Functions](#loss-functions)
    - [Mean Absolute Error (MAE)](#mean-absolute-error-mae)
    - [Mean Squared Error (MSE)](#mean-squared-error-mse)
    - [Root Mean Square Error (RMSE)](#root-mean-square-error-rmse)
- [Optimizers](#optimizers)
    - [Gradient Descent](#gradient-descent)
- [Statistical Pathologies](#statistical-pathologies)
    - [Curse of Dimensionality](#curse-of-dimensionality)

## Core Algorithms

### Linear Regression

Single feature representation:

$\hat{y}​^​=wx+b $
- w = weight
- b = bias

Multiple feature reality:

$\hat{y}​=β_{0}​+β_{1}​x_{1}​+β_{2}​x_{2}​+⋯+β_{p}​x_{p}​$

$
X =
\begin{bmatrix}
1 & x_{11} & x_{12} & \dots & x_{1p} \\
1 & x_{21} & x_{22} & \dots & x_{2p} \\
\vdots & \vdots & \vdots & \ddots & \vdots \\
1 & x_{n1} & x_{n2} & \dots & x_{np}
\end{bmatrix}
$

$
\beta =
\begin{bmatrix}
b \\
w_1 \\
w_2 \\
\vdots \\
w_p
\end{bmatrix}
$

Now the model becomes a single matrix multiplication:

$
\hat{y} = X\beta
$

Where:

- $X \in \mathbb{R}^{n \times (p+1)}$
- $\beta \in \mathbb{R}^{(p+1) \times 1}$
- $\hat{y} \in \mathbb{R}^{n \times 1}$

## Feature Scaling

### Normalization

*Where does this value lie within the observed range?*

|Symbol|Meaning|
|-|-|
|$x$|Original feature value|
|$x_{min}$|Min value in that feature|
|$x_{max}$|Max value in that feature|
|$x'$|Scaled value|

#### Formula:

$$
x' = \frac{x - x_{\min}}{x_{\max} - x_{\min}}
$$

#### Output range:

$$
0 \leq  x'  \leq 1
$$

#### Example:

Credit scores range from 500-800 in dataset. 

500 -> 0 \
650 -> 0.5 \
800 -> 1 

*This approach preserves ranges*

#### When to use:

- Features have known natural bounds
- Distance based algorithms (KNN, K-Means)

### Standardization

*How many standard deviations is this value from the average?*

|Symbol|Meaning|
|-|-|
|$x$|Original feature value|
|$μ$|Feature mean|
|$σ$|Feature standard deviation|
|$x'$|Scaled value|

#### Formula:

$$
x' = \frac{x - \mu}{\sigma}
$$

#### Output distribution:

Mean ≈ 0 
standard deviation ≈ 1

#### Example:

μ = $25,000 \
σ = $10,000

$5,000 -> -2.0 \
$25,000 -> 0.0 \
$45,000 -> 2.0

*This approach does not preserve ranges*

#### When to use:

- Linear regression
- Logistic regression
- SVM
- PCA
- XGBoost
- Features without hard bounds

### Robust Scaling

Ignores extreme tails, great when your data is primarily outliers.

$$
x' = \frac{x - median(x)}{IQR}
$$

$$
IQR = Q_{3}-Q_{1}
$$

### Log Scaling

Transforms exponential distributions into linear distributions.

$$
x' = log(1+x)
$$

## Loss Functions

### Mean Absolute Error (MAE)

*On average, how many units off is my prediction*

**MAE** ($L1$ loss) is easy to interpret and robust to outliers and measured in the same units as your target. Can under penalize outliers.

|Symbol|Meaning|
|-|-|
|$n$|Total number of observations (row count)|
|$y_{i}$|Actual value for observation i|
|$\hat{y}​_{i}$|Predicted value from the model for observation i|

#### Formula:

$$
\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} \left| y_i - \hat{y}_i \right|
$$

### Mean Squared Error (MSE)

*What is the average size of my error when large mistakes are punished aggressively?*

**MSE** ($L2^{2}$ loss) is the core optimization loss by most regression models but the output is in squared units.

|Symbol|Meaning|
|-|-|
|$n$|Total number of observations (row count)|
|$y_{i}$|Actual value for observation i|
|$\hat{y}​_{i}$|Predicted value from the model for observation i|

#### Formula:

$$
\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} \left( y_i - \hat{y}_i \right)^2
$$

### Root Mean Square Error (RMSE)

*What is the typical size of my error, with heavy punishment for big misses?*

**RMSE** ($L2$ loss) is ideal when large errors are very costly.

|Symbol|Meaning|
|-|-|
|$n$|Total number of observations (row count)|
|$y_{i}$|Actual value for observation i|
|$\hat{y}​_{i}$|Predicted value from the model for observation i|
|$(y_{i} - \hat{y}​_{i})^2$|Squared error|

#### Formula:

$$
\text{RMSE} = \sqrt{ \frac{1}{n} \sum_{i=1}^{n} \left( y_i - \hat{y}_i \right)^2 }
$$

## Optimizers

### Gradient Descent

*How does the model actually learn?*

Gradient descent is the process of **iteratively updating model parameters to minimize the loss function**.

For a simple linear model:

$$
\hat{y}_i = wx_i + b
$$

We substitute this into the MSE loss:

$$
\text{MSE}(w,b) = \frac{1}{n} \sum_{i=1}^{n} \left( y_i - (wx_i + b) \right)^2
$$

This converts prediction into a **loss surface over the parameters** \( w \) and \( b \).

---

### Compute the Gradients

We calculate how the loss changes with respect to each parameter.

#### Partial derivative w.r.t weight \( w \)

$$
\frac{\partial \text{MSE}}{\partial w} = -\frac{2}{n} \sum_{i=1}^{n} x_i (y_i - \hat{y}_i)
$$

#### Partial derivative w.r.t bias \( b \)

$$
\frac{\partial \text{MSE}}{\partial b} = -\frac{2}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)
$$

---

### Gradient Vector

The **gradient** is the vector of all partial derivatives:

$$
\nabla L =
\begin{bmatrix}
\frac{\partial L}{\partial w} \\
\frac{\partial L}{\partial b}
\end{bmatrix}
$$

The upside-down triangle \( $\nabla$ \) is called **nabla**.  
It means *“take the derivative with respect to every parameter.”*

This vector points in the direction of **steepest increase in loss** — the slope of the loss surface at your current position.

---

### Update Rule

Using learning rate \( \alpha \):

$$
w := w - \alpha \frac{\partial \text{MSE}}{\partial w}
$$

$$
b := b - \alpha \frac{\partial \text{MSE}}{\partial b}
$$

Equivalently:

$$
\theta := \theta - \alpha \nabla L
$$

This moves the model **downhill on the loss surface**.  
As the gradient approaches zero, the slope flattens — meaning the model is reaching the bottom of the error bowl.

---

### Interpretation

| Term | Meaning |
|------|--------|
| $ y_i - \hat{y}_i $ | Prediction error |
| $ x_i $ | Feature magnitude |
| $ \nabla L $ | Slope of the loss surface |
| Gradient magnitude | How steep the surface is |
| Negative sign | Move in direction of loss decrease |

Gradient descent repeatedly:

1. Predicts  
2. Measures error  
3. Computes gradients  
4. Updates parameters  
5. Repeats until the gradient approaches zero



## Statistical Pathologies

### Curse of Dimensionality