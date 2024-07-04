#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 10:32:55 2024

@author: tosson
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


class SampleEnviromentManager():
           
    def __init__(self, *args, **kwargs):
        super().__init__()



# Define capillary dimensions in micrometers
Lx_um = 2500  # Box length in x-direction (um)
Ly_um = 2500  # Box length in y-direction (um)



def calculate_number_of_particles(volume_um3, concentration1_M, concentration2_M):
    # Avogadro's number
    avogadro_number = 6.022e23  # molecules/mol
    
    # Convert volume from cubic micrometers to liters
    volume_L = volume_um3 * 1e-18
    
    # Calculate the number of molecules of each type
    num_particles1 = concentration1_M * avogadro_number * volume_L
    num_particles2 = concentration2_M * avogadro_number * volume_L
    
    return num_particles1, num_particles2

# Example parameters
volume_um3 = Lx_um*Ly_um*1  # Volume in cubic micrometers
concentration1_M = 1e-6  # Concentration of molecule type 1 in mol/L
concentration2_M = 2e-6  # Concentration of molecule type 2 in mol/L



num_particles1, num_particles2 = calculate_number_of_particles(volume_um3, concentration1_M, concentration2_M)

print(f"Number of particles of type 1: {num_particles1:.2e}")
print(f"Number of particles of type 2: {num_particles2:.2e}")


# Define box dimensions and particle properties in micrometers
Lx_um = 2500  # Box length in x-direction (um)
Ly_um = 2500  # Box length in y-direction (um)
N = 100       # Number of particles
r_mean_um = 20.000  # Mean particle radius (um)
r_std_um = 9.000    # Standard deviation of particle radius (um)
# Convert box dimensions and particle radii to nanometers
Lx = Lx_um * 1000  # Convert um to nm
Ly = Ly_um * 1000  # Convert um to nm
r_mean = r_mean_um * 1000  # Convert um to nm
r_std = r_std_um * 1000    # Convert um to nm

# Define Q-space parameters in 1/nm
qx_min, qx_max, qy_min, qy_max = -0.5, 0.5, -0.5, 0.5
framesize = (1024, 1024)

# Define X-ray wavelength and distance between sample and detector in meters
wavelength_m = 1.0e-9  # Wavelength of X-rays in meters
distance_m = 0.01  # Distance between sample and detector in meters

# Convert wavelength to inverse meters
lambda_inv_m = 1.0 / wavelength_m

# Convert distance to nanometers
distance_nm = distance_m * 1.0e9

# Generate random positions and sizes for particles within the box
np.random.seed(42)  # For reproducibility
particle_positions = np.random.rand(N, 2) * np.array([Lx, Ly])
particle_radii = np.random.normal(r_mean, r_std, size=N)

# Define Q-space parameters as tuples
qx_range = (-3.0, 3.0)
qy_range = (0.1, 3.0)

# Generate Q-space grid in 1/nm
qx = np.linspace(qx_range[0] * 1000, qx_range[1] * 1000, framesize[0])
qy = np.linspace(qy_range[0] * 1000, qy_range[1] * 1000, framesize[1])
Qx, Qy = np.meshgrid(qx, qy)
Q = np.sqrt(Qx**2 + Qy**2)

# Calculate form factor for each particle
form_factors = np.zeros_like(Q)
for i in range(N):
    QR = Q * particle_radii[i]
    form_factors += 3 * ((np.sin(QR) - QR * np.cos(QR)) / (QR**3))**2
    
    
    
foem_f_fft = np.fft.fft2(form_factors)
fft_result_shifted = np.fft.fftshift(foem_f_fft)
xray_frame = np.abs(fft_result_shifted)
plt.imshow(xray_frame, extent=(*qx_range, *qy_range),)
plt.show()

   
plt.figure(figsize=(12, 5))

# Subplot 1: Particles in a box in real space
plt.subplot(121)
plt.scatter(particle_positions[:, 0]*1e-3, particle_positions[:, 1]*1e-3, s=particle_radii*1e-3, alpha=0.5)
plt.xlim(0, Lx_um)
plt.ylim(0, Ly_um)
plt.gca().set_aspect('equal', adjustable='box')
plt.xlabel('x (um)')
plt.ylabel('y (um)')
plt.title('Particles in a Box (real space)')

# Subplot 2: SAXS image in Q-space
plt.subplot(122)
plt.imshow(form_factors, extent=(*qx_range, *qy_range), cmap='jet', norm=LogNorm(), origin='lower')
#plt.imshow(form_factors, extent=(*qx_range, *qy_range), cmap='hot', origin='lower')
plt.colorbar(label='Intensity')
plt.xlabel('qx (1/nm)')
plt.ylabel('qy (1/nm)')
plt.title('SAXS Image in Q-space (log scale)')

plt.show()