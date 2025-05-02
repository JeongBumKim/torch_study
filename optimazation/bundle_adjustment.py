import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import least_squares

# ✅ Ground truth 3D points
points_3d_true = np.array([
    [0.0, 0.0, 5.0],
    [1.0, 0.5, 6.0],
    [-1.0, -0.5, 4.5],
    [0.5, -1.0, 5.5],
    [-0.5, 1.0, 5.2]
])

# ✅ Camera positions (fixed)
cameras = np.array([
    [0.0, 0.0, 0.0],
    [2.0, 0.0, 0.0],
    [-2.0, 0.0, 0.0]
])

# ✅ Camera projection (simple pinhole: x/z, y/z)
def project(points, cam_pos):
    rel = points - cam_pos
    x_proj = rel[:, 0] / rel[:, 2]
    y_proj = rel[:, 1] / rel[:, 2]
    return np.stack((x_proj, y_proj), axis=1)

# ✅ Observations from all cameras
observations = [project(points_3d_true, cam) for cam in cameras]

# ✅ Add noise
observations = [obs + np.random.randn(*obs.shape) * 0.01 for obs in observations]

# ✅ Initial guess: slightly noisy 3D points
points_init = points_3d_true + np.random.randn(*points_3d_true.shape) * 0.2

# ✅ Residuals: reprojection error
def residuals(params):
    points = params.reshape((-1, 3))
    res = []
    for i, cam in enumerate(cameras):
        proj = project(points, cam)
        res.append((proj - observations[i]).ravel())
    return np.concatenate(res)

# ✅ Optimize
result = least_squares(residuals, points_init.ravel(), verbose=1)
points_optimized = result.x.reshape(-1, 3)

# ✅ 3D 시각화
fig = plt.figure(figsize=(10,7))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(*points_3d_true.T, c='g', label='True Points')
ax.scatter(*points_init.T, c='r', label='Initial Guess')
ax.scatter(*points_optimized.T, c='b', label='Optimized Points')

# 카메라 위치
for cam in cameras:
    ax.scatter(*cam, marker='^', c='k', s=100, label='Camera')

ax.set_title("3D Bundle Adjustment (Simulated)")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.legend()
ax.view_init(elev=20, azim=60)
plt.tight_layout()
plt.show()
