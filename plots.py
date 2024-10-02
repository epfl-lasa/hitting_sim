## Distance vs flux

import numpy as np
import matplotlib.pyplot as plt


flux = np.array([0.6, 0.7, 0.8, 0.9, 1.0])

obj1_7 = np.array([0.44, 0.54, 0.64, 0.68, 0.70])
obj2_7 = np.array([0.44, 0.52, 0.56, 0.60, 0.64])
obj3_7 = np.array([0.45, 0.53, 0.59, 0.65, 0.69])

obj1_6 = np.array([0.44, 0.55, 0.62, 0.69, 0.71])
obj2_6 = np.array([0.44, 0.52, 0.57, 0.63, 0.68])
obj3_6 = np.array([0.46, 0.52, 0.57, 0.65, 0.71])

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

ax1.plot(flux, obj1_7, label='Object 1 - Joint 7', marker="*", markersize = 15, linewidth=5, color='r', linestyle=':')
ax1.plot(flux, obj2_7, label='Object 2 - Joint 7', marker="*", markersize = 15, linewidth=5, color='g', linestyle=':')
ax1.plot(flux, obj3_7, label='Object 3 - Joint 7', marker="*", markersize = 15, linewidth=5, color='b', linestyle=':')
ax1.plot(flux, obj1_6, label='Object 1 - Joint 6', marker="o", markersize = 15, linewidth=5, color='r', linestyle='-.')
ax1.plot(flux, obj2_6, label='Object 2 - Joint 6', marker="o", markersize = 15, linewidth=5, color='g', linestyle='-.')
ax1.plot(flux, obj3_6, label='Object 3 - Joint 6', marker="o", markersize = 15, linewidth=5, color='b', linestyle='-.')
# ax1.set_xlabel('Flux (m/s)', fontsize=30)
ax1.set_ylabel('Distance (m)', fontsize=30)
# ax1.xticks(fontsize=15)
# ax1.yticks(fontsize=15)
ax1.tick_params(labelsize=20)
ax1.set(ylim=(0.4, 0.8))
ax1.legend(fontsize=15)

## rebound vs flux

r1_7 = np.array([4.1, 3.7, 3.4, 4.2, 3.8])
r2_7 = np.array([4.1, 3.8, 3.6, 4.2, 3.5])
r3_7 = np.array([4.3, 3.7, 3.1, 3.5, 4.3])
r1_6 = np.array([3.5, 3.2, 3.0, 3.5, 3.2])
r2_6 = np.array([3.7, 3.5, 3.3, 3.0, 3.2])
r3_6 = np.array([3.7, 3.2, 2.8, 3.1, 3.9])

ax2.plot(flux, r1_7, label='Object 1 - Joint 7', marker="*", markersize = 15, linewidth=5, color='r', linestyle=':')
ax2.plot(flux, r2_7, label='Object 2 - Joint 7', marker="*", markersize = 15, linewidth=5, color='g', linestyle=':')
ax2.plot(flux, r3_7, label='Object 3 - Joint 7', marker="*", markersize = 15, linewidth=5, color='b', linestyle=':')
ax2.plot(flux, r1_6, label='Object 1 - Joint 6', marker="o", markersize = 15, linewidth=5, color='r', linestyle='-.')
ax2.plot(flux, r2_6, label='Object 2 - Joint 6', marker="o", markersize = 15, linewidth=5, color='g', linestyle='-.')
ax2.plot(flux, r3_6, label='Object 3 - Joint 6', marker="o", markersize = 15, linewidth=5, color='b', linestyle='-.')
ax2.set(ylim=(2.5, 5))
ax2.set_xlabel('Flux (m/s)', fontsize=30)
ax2.set_ylabel('Relative rebound (%)', fontsize=30)
# ax2.xticks(fontsize=15)
# ax2.yticks(fontsize=15)
# plt.legend(fontsize=15)
ax2.tick_params(labelsize=20)
plt.show()

