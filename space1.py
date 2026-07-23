import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define the exact path to your local zeros dataset
csv_path = "/Users/karlroesch/Desktop/karl/VSworking/RZ/first100000.csv"

# Load the Riemann zeros directly from your file
if os.path.exists(csv_path):
    try:
        gamma = np.loadtxt(csv_path, max_rows=100)
        print(f"Successfully loaded {len(gamma)} zeros from {csv_path}")
    except Exception as e:
        print(f"Error loading CSV, falling back to array. Details: {e}")
        gamma = np.array([14.134725142, 21.022039639, 25.010857580, 30.424876126])
else:
    print(f"File not found at {csv_path}. Using standard Riemann zeros array.")
    gamma = np.array([14.134725142, 21.022039639, 25.010857580, 30.424876126])

# DYNAMIC PARAMETER RANGE: 
# Extend 'a' slightly past the maximum loaded zero so all points can be mapped.
max_zero = np.max(gamma)
a = np.linspace(0, max_zero + 10, 50000) # Increased resolution for the wider range

# Calculate winding phases using the first 3 dimensions for the T^3 projection
theta1 = (np.pi * a) / gamma[0]
theta2 = (np.pi * a) / gamma[1]
theta3 = (np.pi * a) / gamma[2]

# Nesting radii to project the multi-dimensional torus winding into 3D space
R1 = 4.0
R2 = 1.5
R3 = 0.6

# Project the irrational Kronecker winding to 3D Cartesian coordinates
x = (R1 + R2 * np.cos(theta2) + R3 * np.cos(theta3)) * np.cos(theta1)
y = (R1 + R2 * np.cos(theta2) + R3 * np.cos(theta3)) * np.sin(theta1)
z = R2 * np.sin(theta2) + R3 * np.sin(theta3)

# Calculate the multi-dimensional filter value F(a) over ALL loaded zeros
F_a = np.ones_like(a)
for i in range(len(gamma)):
    F_a *= np.cos(np.pi * a / gamma[i])**2

# Create the visualization plots
fig = plt.figure(figsize=(16, 7))

# Panel 1: 3D Torus Kronecker Winding
ax1 = fig.add_subplot(121, projection='3d')
ax1.plot(x, y, z, color='blue', alpha=0.3, lw=0.5, label="State Trajectory")

# Mark the locations where parameter 'a' equals ALL loaded Riemann zero values
# Using a slightly smaller size (s=20) so 100 points don't overcrowd the line
for zero_val in gamma:
    idx = np.argmin(np.abs(a - zero_val))
    ax1.scatter(x[idx], y[idx], z[idx], color='red', s=20, zorder=5)

ax1.set_title(f"Projected Hilbert ")
ax1.set_xlabel("X-Axis")
ax1.set_ylabel("Y-Axis")
ax1.set_zlabel("Z-Axis")
ax1.view_init(elev=20, azim=55)

# Panel 2: The Coherence Filter F(a) showing the peak resonances
ax2 = fig.add_subplot(122)
ax2.plot(a, F_a, color='purple', lw=1.2, label=f'$F(a)$ (Product over {len(gamma)} Zeros)')

# Draw dashed red lines at the exact positions of ALL loaded Riemann zeros
for zero_val in gamma:
    ax2.axvline(x=zero_val, color='red', linestyle='--', alpha=0.25, lw=0.8)

ax2.set_title(f"Peaks $F(a)$\nTotal Aligned Resonance Fields: {len(gamma)}")
ax2.set_xlabel("Coordinate Parameter $a$")
ax2.set_ylabel("Filter Transmission")
ax2.grid(True, alpha=0.3)
ax2.legend()

plt.tight_layout()
plt.show()