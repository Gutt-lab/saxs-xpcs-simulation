#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 10:03:14 2024

@author: tosson
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tqdm import tqdm

from joblib import Parallel, delayed

import skbeam.core.roi as roi
import skbeam.core.utils as utils
import skbeam.core.correlation as corr

from scipy.signal import correlate



particles = []
def update(frame):
    global particles
    particles = update_particles(particles, D, time_step, Lx, Ly)
    scat.set_offsets(particles)
    time_text.set_text('Time = {:.2f} s'.format(frame * time_step))
    return scat, time_text
# Define a function to generate a single frame
def generate_frame(frame_index):
    # Update particle positions
    updated_particles = update_particles(particles, D, time_step, Lx, Ly)
        
    # Calculate SAXS image using FFT
    saxs_image = calculate_saxs_fft(updated_particles, radii, Lx, Ly)
    
    return saxs_image


# Function to initialize particle positions and radii
def initialize_particles(num_particles, box_length_x, box_length_y, r_mean, r_std):
    radii = np.random.normal(r_mean, r_std, num_particles)
    for _ in range(num_particles):
        x = np.random.uniform(max(radii), box_length_x - max(radii))
        y = np.random.uniform(max(radii), box_length_y - max(radii))
        particles.append([x, y])
    return np.array(particles), radii

# Function to update particle positions based on Brownian motion
def update_particles(particles, diffusion_coefficient, time_step, box_length_x, box_length_y):
    displacement = np.sqrt(2 * diffusion_coefficient * time_step) * np.random.randn(len(particles), 2)
    particles += displacement
    
    # Apply periodic boundary conditions
    particles[:, 0] = np.mod(particles[:, 0], box_length_x)
    particles[:, 1] = np.mod(particles[:, 1], box_length_y) 
    return particles


def calculate_saxs_fft(particles, radii, Lx, Ly, q_resolution=201):
    # Define coordinates in real space
    x = np.linspace(0, Lx, q_resolution)
    y = np.linspace(0, Ly, q_resolution)
    xx, yy = np.meshgrid(x, y)

    # Compute scattering intensity distribution in real space
    intensity_distribution = np.zeros((q_resolution, q_resolution))
    for particle, radius in zip(particles, radii):
        r_squared = (xx - particle[0])**2 + (yy - particle[1])**2
        intensity_distribution += np.where(r_squared <= radius**2, 1, 0)

    # Perform 2D FFT
    fft_result = np.fft.fft2(intensity_distribution)

    # Apply FFT shift
    fft_result_shifted = np.fft.fftshift(fft_result)

    # Calculate magnitude of the FFT result (spectrum)
    saxs_spectrum = np.abs(fft_result_shifted)

    return saxs_spectrum
    
# Calculate g2 function for each ring
def calculate_g2_for_each_ring(frames, ring_mask):
    # Extract frame size
    num_frames, framesize, _ = frames.shape
    num_rings = np.max(ring_mask)
    
    # Initialize arrays to store g2 functions for each ring
    g2_ring = np.zeros((num_rings, num_frames))
    
    # Calculate the intensity profiles for each ring
    for ring in range(1, num_rings + 1):
        ring_indices = np.where(ring_mask == ring)
        num_pixels = len(ring_indices[0])
        intensity_profiles = np.zeros((num_frames, num_pixels))
        for i in range(num_frames):
            intensity_profiles[i] = frames[i][ring_indices]
        
        # Calculate the autocorrelation function for each ring
        acf = correlate(intensity_profiles, intensity_profiles, mode='same', method='fft')
        
        # Calculate the mean intensity for each frame
        mean_intensity = np.mean(intensity_profiles, axis=1)
        
        # Reshape mean intensity to match the shape of acf
        mean_intensity_square = mean_intensity[:, np.newaxis, np.newaxis] * mean_intensity[:, np.newaxis, np.newaxis]
       
        # Normalize by the square of the mean intensity
        g2 = acf / mean_intensity_square
        
        g2_ring[ring - 1] = g2.diagonal(offset=0, axis1=1, axis2=2).mean(axis=1)
        
    return g2_ring

Lx = 400  # Box length in x-direction (um)
Ly = 400  # Box length in y-direction (um)
N  = 50   # Number of particles
r_mean = 5.00  # Mean particle radius (um)
r_std  = 1.0   # Standard deviation of particle radius (um)

# Initialize particle positions and radii
particles, radii = initialize_particles(N, Lx, Ly, r_mean, r_std)


plt.figure(figsize=(6, 6))
plt.scatter(particles[:, 0], particles[:, 1], s=np.pi * (radii ** 2), alpha=0.5)
plt.xlim(0, Lx)
plt.ylim(0, Ly)
plt.xlabel('X ($\mu$m)')
plt.ylabel('Y ($\mu$m)')
plt.title('Box (Initial State)', fontsize = 12)
plt.gca().set_aspect('equal', adjustable='box')
plt.show()


# Calculate SAXS spectrum using FFT
saxs_spectrum = calculate_saxs_fft(particles, radii, Lx, Ly) # q_resolution=100

cmap = "jet" # 'viridis'

# Plot SAXS spectrum
plt.figure(figsize=(8, 6))
plt.imshow(np.log(1 + saxs_spectrum), cmap=cmap, origin='lower', extent=(-np.pi/Lx*1e3, np.pi/Lx*1e3, -np.pi/Ly*1e3, np.pi/Ly*1e3))
plt.colorbar(label='Log Intensity')
plt.xlabel('$q_x$ ($nm^{-1}$)')
plt.ylabel('$q_y$ ($nm^{-1}$)')
plt.title('X-ray Scattering Image (SAXS) - FFT')
plt.show()



D_0 = 6.0e-10      # Diffusion coefficient [m^2/s]
D = D_0*1e12       # Diffusion coefficient [um^2/s]
num_frames = 1000  # Number of frames to simulate
time_step = 0.001  # Time step for simulation [s]


# Set up the figure
fig, ax = plt.subplots()
ax.set_xlim(0, Lx)
ax.set_ylim(0, Ly)
ax.set_aspect('equal')
plt.xlabel('X ($\mu$m)')
plt.ylabel('Y ($\mu$m)')
scat = ax.scatter(particles[:, 0], particles[:, 1], s=np.pi * radii**2)
time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)

# Create animation
ani = animation.FuncAnimation(fig, update, frames=num_frames, interval=50, blit=True)

# Save animation as a video file
ani.save('animation.mp4', writer='ffmpeg', fps=30)

plt.show()

print("Done!")


# Generate frames in parallel
frames = Parallel(n_jobs=-1)(delayed(generate_frame)(i) for i in tqdm(range(num_frames)))

# Convert frames to numpy array
frames = np.array(frames)

cmap = "jet" # 'viridis'

# Plot SAXS spectrum
plt.figure(figsize=(8, 6))
plt.imshow(np.log(1 + frames[10]), cmap=cmap, origin='lower', extent=(-np.pi/Lx*1e3, np.pi/Lx*1e3, -np.pi/Ly*1e3, np.pi/Ly*1e3))
plt.colorbar(label='Log Intensity')
plt.xlabel('$q_x$ ($nm^{-1}$)')
plt.ylabel('$q_y$ ($nm^{-1}$)')
plt.title('X-ray Scattering Image (SAXS) - FFT')
plt.show()

# Plot SAXS spectrum
plt.figure(figsize=(8, 6))
plt.imshow(frames[10], cmap=cmap, origin='lower')
plt.colorbar(label='Log Intensity')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('X-ray Scattering Image (SAXS) - FFT')
plt.show()


inner_radius = 10 # radius of the inner ring in pixels
width        =  8 # width in pixel of the q-rings
spacing      =  1 # spacing between q-rings in pixel
num_rings    = 10 # number of rings

imshape = frames[0].shape

# Specify the center position
y_center, x_center = frames[0].shape[0] // 2, frames[0].shape[1] // 2
center = (y_center, x_center)

xlim = (x_center-inner_radius-num_rings*(spacing+width),
        x_center+inner_radius+num_rings*(spacing+width))
ylim = (y_center-inner_radius-num_rings*(spacing+width),
        y_center+inner_radius+num_rings*(spacing+width))

edges = roi.ring_edges(inner_radius, width, spacing, num_rings) #calculate edges of rings
rings = roi.rings(edges, center, imshape)

ring_mask = rings # without beamstop


# Plot SAXS spectrum
plt.figure(figsize=(8, 6))
plt.imshow(ring_mask, cmap=cmap, origin='lower')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Rings')
plt.show()

num_levels=1
num_bufs  = 100

g2, lag_steps = corr.multi_tau_auto_corr(num_levels=num_levels, num_bufs=num_bufs, labels=ring_mask, images=frames)
lag_steps =lag_steps*time_step # tau axis for g2 array

print("Done!")


num_ring = 1
plt.figure(figsize=(8, 6))
plt.plot(lag_steps,g2[:,num_ring],marker='.',linestyle='',label='{:.3f}'.format(num_ring))
plt.xscale('log')
plt.xlabel(r'$\tau$ (s)')
plt.ylabel(r'$g^{(2)}$($\tau$,q)')
#plt.ylim([1.0, 1.01])
plt.show()


# Calculate g2 function for each ring
g2_ring = calculate_g2_for_each_ring(frames, ring_mask)

# Generate lag times
lag_times = np.arange(-len(frames)//2, len(frames)//2)

tau = time_step * lag_times

# Plot g2 function for each ring
plt.figure(figsize=(8, 10))

for i in range(num_rings):
    plt.plot(tau[tau>0], g2_ring[i][tau>0],marker='.', label=f'Ring {i+1}')
plt.xlabel('Lag Time, s')
plt.ylabel('g2 Function')
plt.title('Intensity Autocorrelation (g2) for Each Ring')
plt.xscale('log')
plt.legend()
plt.show()

