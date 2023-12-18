import numpy as np
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def transform(params, x):
    theta_x, theta_y, theta_z, tx, ty, tz, sx, sy, sz = params
    Rx = np.array([
        [1, 0, 0, 0],
        [0, np.cos(theta_x), -np.sin(theta_x), 0],
        [0, np.sin(theta_x), np.cos(theta_x), 0],
        [0, 0, 0, 1]
    ])
    Ry = np.array([
        [np.cos(theta_y), 0, np.sin(theta_y), 0],
        [0, 1, 0, 0],
        [-np.sin(theta_y), 0, np.cos(theta_y), 0],
        [0, 0, 0, 1]
    ])
    Rz = np.array([
        [np.cos(theta_z), -np.sin(theta_z), 0, 0],
        [np.sin(theta_z), np.cos(theta_z), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    T = np.array([
        [1, 0, 0, tx],
        [0, 1, 0, ty],
        [0, 0, 1, tz],
        [0, 0, 0, 1]
    ])
    S = np.diag([sx, sy, sz, 1])
    M = np.dot(np.dot(np.dot(T, Rz), Ry), Rx)
    M = np.dot(M, S)
    x_homogeneous = np.column_stack([x, np.ones(x.shape[0])])
    transformed_x_homogeneous = np.dot(x_homogeneous, M.T)
    return transformed_x_homogeneous[:, :3]

def error(params, x, y):
    transformed_x = transform(params, x)
    return np.linalg.norm(transformed_x - y, axis=1)

# Example points for camera and LiDAR (3D)
x = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15], [16, 17, 18], [19, 20, 21], [22, 23, 24], [25, 26, 27]])
y = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9], [1.0, 1.1, 1.2], [1.3, 1.4, 1.5], [1.6, 1.7, 1.8], [1.9, 2.0, 2.1], [2.2, 2.3, 2.4], [2.5, 2.6, 2.7]])

# Initial guess for the transformation parameters
initial_params = [0, 0, 0, 0, 0, 0, 1, 1, 1]

# Optimization
result = least_squares(error, initial_params, args=(x, y), method='lm')
optimized_params = result.x
transformed_x = transform(optimized_params, x)

# Error calculation
difference = y - transformed_x
mse = np.mean(np.sum(difference**2, axis=1))
rmse = np.sqrt(mse)

# Results
print("Optimized Parameters:", optimized_params)
print("Root Mean Squared Error (RMSE):", rmse)

# 3D Plotting
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(y[:, 0], y[:, 1], y[:, 2], label='Lidar Points', c='blue', marker='o')
ax.scatter(transformed_x[:, 0], transformed_x[:, 1], transformed_x[:, 2], label='Transformed Camera Points', c='red', marker='x')
ax.legend()
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Z-axis')
plt.title('Alignment between Lidar and Transformed Camera Points in 3D')
plt.show()
