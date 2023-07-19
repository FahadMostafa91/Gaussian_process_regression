import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

# Define the true function and generate synthetic data
def true_func(X):
    return np.sin(3*X) + np.cos(5*X)

X = np.linspace(0, 1, 20)[:, np.newaxis]
y = true_func(X) + 0.1*np.random.randn(20)[:, np.newaxis]

# Define the kernel function and fit the Gaussian process model
kernel = C(1.0, (1e-3, 1e3)) * RBF(0.1, (1e-2, 1e2))
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
gp.fit(X, y)

# Generate predictions and plot the results
X_pred = np.linspace(0, 1, 100)[:, np.newaxis]
y_pred, std_pred = gp.predict(X_pred, return_std=True)

plt.figure(figsize=(10, 5))
plt.plot(X, y, 'r.', markersize=10, label='Observations')
plt.plot(X_pred, true_func(X_pred), 'b-', label='True function')
plt.plot(X_pred, y_pred, 'k-', label='Predicted function')
plt.fill_between(X_pred[:, 0], y_pred[:, 0] - 1.96*std_pred, y_pred[:, 0] + 1.96*std_pred, alpha=0.2)
plt.xlabel('Input variable (X)')
plt.ylabel('Output variable (y)')
plt.title('Gaussian Process Regression')
plt.legend(loc='upper left')
plt.show()

"""
Here, we first define the true function as a sine wave with a period of 1. 
We then generate 20 synthetic data points by randomly sampling x values between 0 and 1 
and adding Gaussian noise to the true function values. We then define the kernel for 
Gaussian process regression as an RBF kernel with length scale 1 and a white noise kernel 
with noise level 1. We create a Gaussian process regression model with this kernel and
 fit it to the synthetic data. We then generate a set of test points to make predictions 
 and use the fitted model to make predictions for these points. Finally, we plot the true
 function, the synthetic data, and the predictions along with their standard deviations.
 """
 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

# Define the true function
def true_function(x):
    return np.sin(2 * np.pi * x)

# Generate synthetic data
n_samples = 20
rng = np.random.RandomState(1234)
X = rng.uniform(0, 1, n_samples)[:, np.newaxis]
y = true_function(X) + rng.normal(scale=0.1, size=n_samples)[:, np.newaxis]

# Define the kernel for Gaussian process regression
kernel = RBF(length_scale=1.0, length_scale_bounds=(1e-1, 10.0)) + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-3, 1e3))

# Define the Gaussian process regression model
gp = GaussianProcessRegressor(kernel=kernel, alpha=0.0)

# Fit the model to the data
gp.fit(X, y)

# Generate test data for predictions
x_test = np.linspace(0, 1, 100)[:, np.newaxis]

# Make predictions with the fitted model
y_pred, y_std = gp.predict(x_test, return_std=True)

# Plot the true function, the training data, and the predictions
plt.plot(x_test, true_function(x_test), 'r:', label=r'$\sin(2\pi x)$')
plt.plot(X, y, 'r.', markersize=10, label='Observations')
plt.plot(x_test, y_pred, 'b-', label='Predictions')
plt.fill_between(x_test[:, 0], y_pred[:, 0] - y_std, y_pred[:, 0] + y_std, alpha=0.2, color='blue')
plt.xlabel('x')
plt.ylabel('y')
plt.ylim(-1.5, 1.5)
plt.legend(loc='upper left')
plt.show()

 
 
 
 
 
 
 
 
 
 