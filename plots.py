## Distance vs flux

import numpy as np
import matplotlib.pyplot as plt


flux = np.array([0.6, 0.7, 0.8, 0.9, 1.0])

obj1_7 = np.array([0.44, 0.54, 0.64, 0.68, 0.70])
obj2_7 = np.array([0.44, 0.52, 0.56, 0.60, 0.64])
obj3_7 = np.array([0.45, 0.53, 0.59, 0.65, 0.69])

obj1_6 = np.array([0.44, 0.55, 0.62, 0.69, 0.71])
obj2_6 = np.array([0.44, 0.52, 0.57, 0.61, 0.65])
obj3_6 = np.array([0.46, 0.52, 0.57, 0.65, 0.71])

plt.plot(flux, obj1_7, label='Object 1 - Joint 7')
plt.plot(flux, obj2_7, label='Object 2 - Joint 7')
plt.plot(flux, obj3_7, label='Object 3 - Joint 7')

plt.plot(flux, obj1_6, label='Object 1 - Joint 6')
plt.plot(flux, obj2_6, label='Object 2 - Joint 6')
plt.plot(flux, obj3_6, label='Object 3 - Joint 6')
plt.xlabel('Flux')
plt.ylabel('Distance')
plt.legend()
plt.show()

## rebound vs flux

r1_7 = np.array([0.44, 0.54, 0.64, 0.68, 0.70])
r2_7 = np.array([0.44, 0.52, 0.56, 0.60, 0.64])
r3_7 = np.array([0.45, 0.53, 0.59, 0.65, 0.69])

r1_6 = np.array([0.44, 0.55, 0.62, 0.69, 0.71])
r2_6 = np.array([0.44, 0.52, 0.57, 0.61, 0.65])
r3_6 = np.array([0.46, 0.52, 0.57, 0.65, 0.71])

plt.plot(flux, r1_7, label='Object 1 - Joint 7')
plt.plot(flux, r2_7, label='Object 2 - Joint 7')
plt.plot(flux, r3_7, label='Object 3 - Joint 7')

plt.plot(flux, r1_6, label='Object 1 - Joint 6')
plt.plot(flux, r2_6, label='Object 2 - Joint 6')
plt.plot(flux, r3_6, label='Object 3 - Joint 6')
plt.xlabel('Flux')
plt.ylabel('Rebound')
plt.legend()
plt.show()