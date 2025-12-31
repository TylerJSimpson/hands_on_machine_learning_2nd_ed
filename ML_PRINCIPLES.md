# Machine Learning Principles

## Appendix
- [Loss Functions](#loss-functions)
    - [Mean Absolute Error (MAE)](#mean-absolute-error-mae)
    - [Root Mean Square Error (RMSE)](#root-mean-square-error-rmse)
- [Feature Scaling](#feature-scaling)
    - [Normalization](#normalization)
    - [Standardization](#standardization)
    - [Robust Scaling](#robust-scaling)
    - [Log Scaling](#log-scaling)
- [Statistical Pathologies](#statistical-pathologies)
    - [Curse of Dimensionality](#curse-of-dimensionality)

## Loss Functions

### Mean Absolute Error (MAE)

*On average, how many units off is my prediction*

**MAE** is easy to interpret and robust to outliers and measured in the same units as your target. Can under penalize outliers.

|Symbol|Meaning|
|-|-|
|$n$|Total number of observations (row count)|
|$y_{i}$|Actual value for observation i|
|$\hat{y}​_{i}$|Predicted value from the model for observation i|

#### Formula:

$$
\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} \left| y_i - \hat{y}_i \right|
$$


### Root Mean Square Error (RMSE)

*What is the typical size of my error, with heavy punishment for big misses?*

**RMSE** is ideal when large errors are very costly.

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

## Statistical Pathologies

### Curse of Dimensionality