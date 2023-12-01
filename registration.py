import numpy as np
from scipy.optimize import least_squares
import matplotlib.pyplot as plt


'''
Apply a 2D transformation to the points in x.

Parameters:
params: List of transformation parameters [theta, tx, ty, sx, sy]
x: 2D array of points to be transformed

Returns:
transformed_x: 2D array of transformed points
'''

def transform(params, x):

    theta, tx, ty, sx, sy = params
    T = np.array([
        [sx * np.cos(theta), -sy * np.sin(theta), tx],
        [sx * np.sin(theta), sy * np.cos(theta), ty],
        [0, 0, 1]
    ])
    x_homogeneous = np.column_stack([x, np.ones(x.shape[0])])
    transformed_x_homogeneous = np.dot(x_homogeneous, T.T)
    return transformed_x_homogeneous[:, :2]


'''
Compute the error between transformed x and y.

Parameters:
params: Transformation parameters
x: Original points
y: Target points to align with

Returns:
Error as the L2 norm between transformed x and y
'''
def error(params, x, y):
    transformed_x = transform(params, x)
    return np.linalg.norm(transformed_x - y, axis=1)

# Testing points for camera and LiDAR
x = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]])
y = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8], [0.9, 1.0], [1.1, 1.2]])

# Initial guess for the transformation parameters
initial_params = [0, 0, 0, 1, 1]

# Perform optimization to find the best transformation parameters
result = least_squares(error, initial_params, args=(x, y), method='lm')
optimized_params = result.x
transformed_x = transform(optimized_params, x)

difference = y - transformed_x
mse = np.mean(np.sum(difference**2, axis=1))
rmse = np.sqrt(mse)


'''
BELOW is for visual purposes only.
'''

print("Optimized Parameters (theta, tx, ty, sx, sy):", optimized_params)
print("Root Mean Squared Error (RMSE):", rmse)

plt.scatter(y[:, 0], y[:, 1], label='Lidar Points', c='blue', marker='o')
plt.scatter(transformed_x[:, 0], transformed_x[:, 1], label='Transformed Camera Points', c='red', marker='x')
plt.legend()
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Alignment between Lidar and Transformed Camera Points')
plt.grid(True)
plt.show()
