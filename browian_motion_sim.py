import numpy as np
import matplotlib.pyplot as plt

# Constants
k_B = 1.380649e-23  # Boltzmann constant in J/K
T = 300  # Temperature in Kelvin
eta = 1e-3  # Viscosity of water in Pa.s
r = 1e-6  # Radius of the particle in meters
gamma = 6 * np.pi * eta * r  # Drag coefficient
D = k_B * T / gamma  # Diffusion coefficient
dt = 0.01  # Time step in seconds
num_steps = 10000  # Number of steps in the simulation

# Initialize arrays to store the positions
x = np.zeros(num_steps)
y = np.zeros(num_steps)

# Initial velocity (assuming it's zero)
vx = 0
vy = 0

# Simulation loop
for i in range(1, num_steps):
    # Random force components
    eta_x = np.sqrt(2 * D / dt) * np.random.normal()
    eta_y = np.sqrt(2 * D / dt) * np.random.normal()
    
    # Update velocities
    vx += (-gamma * vx + eta_x) * dt
    vy += (-gamma * vy + eta_y) * dt
    
    # Update positions
    x[i] = x[i-1] + vx * dt
    y[i] = y[i-1] + vy * dt

# Plot the trajectory
plt.figure(figsize=(8, 6))
plt.plot(x, y, lw=0.5)
plt.title('Brownian Motion of a Particle in 2D')
plt.xlabel('x position (m)')
plt.ylabel('y position (m)')
plt.grid(True)
plt.show()