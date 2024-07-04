#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 09:35:17 2024

@author: tosson
"""
import asyncio
from threading import Thread

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Rectangle

from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import matplotlib.cm as cm
from joblib import Parallel, delayed

import skbeam.core.roi as roi
import skbeam.core.utils as utils
import skbeam.core.correlation as corr

from scipy.signal import correlate

from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf

particles = []

# sample-wise details
Lx = 1000  # Box length in x-direction (um)
Ly = 1000  # Box length in y-direction (um)
N  = 50  # Number of particles
r_mean = 5.00  # Mean particle radius (um)
r_std  = 0.5   # Standard deviation of particle radius (um)
k_B = 1.380649e-23  # Boltzmann constant in J/K
T = 273  # Temperature in Kelvin
eta = 19e-3 # Viscosity of buffer in Pa.s
r = 1e-9  # Radius of the particle in meters
gamma = 6 * np.pi * eta * r  # Drag coefficient
D = k_B * T / gamma  # Diffusion coefficient
D = D*1e12   
#D_0 = 6.0e-10      # Diffusion coefficient [m^2/s]
#D = D_0*1e12       # Diffusion coefficient [um^2/s]

lab_time = 100.0 # [s]
sample_time_step = 0.1 # The dynamic time [s]
scene_animation_time_step = 0.1
num_scenes = int(lab_time/sample_time_step)    # Number of steps 

x_beam_size = 200 #[um]
y_beam_size = 200 #[um]


#exp-wise details
  
time_step_init = 0.5 # Time step for simulation [s]
num_frames_init = int(lab_time/time_step_init)  # Number of frames to simulate
xray_exposure_scene_step_init = int(num_scenes/num_frames_init)
#detector-wise
x_num_pixel = 1000
y_num_pixel = 500
x_pixel_size = 75e-6 
y_pixel_size = 75e-6  
inner_radius = 70 # radius of the inner ring in pixels
width        =  10 # width in pixel of the q-rings
spacing      =  10 # spacing between q-rings in pixel
num_rings    = 10 # number of rings



# Function to initialize particle positions and radii
def initialize_particles(num_particles, box_length_x, box_length_y, r_mean, r_std):
    radii = np.random.normal(r_mean, r_std, num_particles)
    for _ in range(num_particles):
        x = np.random.uniform(max(radii), box_length_x - max(radii))
        y = np.random.uniform(max(radii), box_length_y - max(radii))
        particles.append([x, y])
    return np.array(particles), radii

def update_particles(s_particles, diffusion_coefficient, time_step, box_length_x, box_length_y):
    displacement = np.sqrt(2 * diffusion_coefficient * time_step)  * np.random.normal(size=[len(s_particles), 2])
    s_particles += displacement
    # Apply periodic boundary conditions
    s_particles[:, 0] = np.mod(s_particles[:, 0], box_length_x)
    s_particles[:, 1] = np.mod(s_particles[:, 1], box_length_y) 
    return s_particles


def update_particles_v2(s_particles, decay_factor, diffusion_coefficient, time_step, box_length_x, box_length_y):
    displacement = np.sqrt(2 * diffusion_coefficient * time_step)  * np.random.normal(size=[len(s_particles), 2]) 
    displacement = displacement * decay_factor
    s_particles += displacement
    # Apply periodic boundary conditions
    s_particles[:, 0] = np.mod(s_particles[:, 0], box_length_x)
    s_particles[:, 1] = np.mod(s_particles[:, 1], box_length_y) 
    return s_particles

def simulate_noise_2d_detector(shape, noise_type='gaussian', mean=0, std_dev=3, lam=10):
    if noise_type == 'gaussian':
        noise = np.random.normal(mean, std_dev, shape)
    elif noise_type == 'poisson':
        noise = np.random.poisson(lam, shape)
    else:
        raise ValueError("Unsupported noise type. Use 'gaussian' or 'poisson'.")
    
    return noise


def generate_xray_frames(particles, particles_radius, box_length_x, box_length_y, x_pixel_size, y_pixel_size, x_pixel_num, y_pixel_num, add_noise = True):

    x = np.linspace(int(Lx/2)-int(x_beam_size/2), int(Lx/2)+int(x_beam_size/2), x_num_pixel)
    y = np.linspace(int(Ly/2)-int(y_beam_size/2), int(Ly/2)+int(y_beam_size/2), y_num_pixel)
    xx, yy = np.meshgrid(x, y)   
    intensity_distribution = np.zeros((y_num_pixel, x_num_pixel))
    for particle, radius in zip(particles, radii):
        r_squared = (xx - particle[0])**2 + (yy - particle[1])**2
        intensity_distribution += np.where(r_squared <= radius**2, 1, 0)
    fft_result = np.fft.fft2(intensity_distribution)

    # Apply FFT shift
    fft_result_shifted = np.fft.fftshift(fft_result)

    # Calculate magnitude of the FFT result (spectrum)
    xray_frame = np.abs(fft_result_shifted) 
    y_center, x_center = int(x_num_pixel // 2), int(y_num_pixel// 2)
    
    # Calculate SAXS image using FFT
    #saxs_image = calculate_saxs_fft(updated_particles, radii, Lx, Ly)
    if add_noise:
        xray_frame = xray_frame + simulate_noise_2d_detector((y_num_pixel, x_num_pixel))
        
    xray_frame[x_center-25:x_center+25, y_center-25:y_center+25] = 0    
    return xray_frame

def generate_q_rings_mask(imshape, num_q_rings, q_inner_radius, q_rings_spacing, q_ring_width):
    # Specify the center position
    y_center, x_center = int(imshape[0] // 2), int(imshape[1] // 2)
    center = (y_center, x_center)

    xlim = (x_center-q_inner_radius-num_q_rings*(q_rings_spacing+q_ring_width),
            x_center+q_inner_radius+num_q_rings*(q_rings_spacing+q_ring_width))
    ylim = (y_center-q_inner_radius-num_q_rings*(q_rings_spacing+q_ring_width),
            y_center+q_inner_radius+num_q_rings*(q_rings_spacing+q_ring_width))
    edges = roi.ring_edges(q_inner_radius, q_ring_width, q_rings_spacing, num_q_rings) #calculate edges of rings
    rings = roi.rings(edges, center, imshape)
    cmap = "jet" # 'viridis'
    plt.figure(figsize=(8, 6))
    plt.imshow(rings, cmap=cmap, origin='lower')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Rings')
    plt.show()
    return rings

def extract_intensity_for_q_rings(frames_arr, q_rings):
    num_frames, framesize, _ = frames_arr.shape
    num_rings = np.max(q_rings)
    intensity_fluctuation = []
    # Initialize arrays to store g2 functions for each ring
    g2_ring = np.zeros((num_rings, num_frames))
    for ring in range(1, num_rings + 1):
        ring_indices = np.where(q_rings == ring)
        num_pixels = len(ring_indices[0])
        intensity_profiles = np.zeros((num_frames, num_pixels))
        for i in range(num_frames):
            intensity_profiles[i] = frames_arr[i][ring_indices]
        intensity_fluctuation.append(np.mean(intensity_profiles, axis = 1))
    return intensity_fluctuation 

def g2_calculation_AT(frames_images, q_rings_mask_arr):
    num_frames, framesize, _ = frames_images.shape
    num_frames = 10
    num_rings = np.max(q_rings_mask_arr)
    
    # Initialize arrays to store g2 functions for each ring
    g2_ring = np.zeros((num_rings, num_frames))
    int_all = []
    # Calculate the intensity profiles for each ring
    for ring in range(1, num_rings + 1):
        ring_indices = np.where(q_rings_mask_arr == ring)
        num_pixels = len(ring_indices[0])
        intensity_profiles = np.zeros((num_frames, num_pixels))
        for i in range(num_frames):
            intensity_profiles[i] = frames_images[i][ring_indices]
        int_all.append(intensity_profiles)
        # Calculate the autocorrelation function for each ring
        acf = correlate(intensity_profiles, intensity_profiles, mode='same', method='fft')
        
        # Calculate the mean intensity for each frame
        mean_intensity = np.mean(intensity_profiles, axis=1)
        
        # Reshape mean intensity to match the shape of acf
        mean_intensity_square = mean_intensity[:, np.newaxis, np.newaxis] * mean_intensity[:, np.newaxis, np.newaxis]
       
        # Normalize by the square of the mean intensity
        g2 = acf / mean_intensity_square
        
        g2_ring[ring - 1] = g2.diagonal(offset=0, axis1=1, axis2=2).mean(axis=1)
        
    return g2_ring, int_all

def g2_calculation(frames_img, ring_mask, plot_g2 = True):
    print(len(frames_img))
    num_levels=4
    num_bufs  = int(len(frames_img)/4)
    if (num_bufs%2)>0:
        num_bufs+= 1 
    #num_bufs = 8
    g2, lag_steps = corr.multi_tau_auto_corr(num_levels=num_levels, num_bufs=num_bufs, labels=ring_mask, images=frames_img)
    if plot_g2:   
        for i in range(g2.shape[1]):
            plt.plot(lag_steps[1:],g2[1:,i],label='Q_ring number: {n}'.format(n=i+1))
            #plt.xscale('log')
            plt.xlabel(r'$\tau$ (s)')
            plt.ylabel(r'$g^{(2)}$($\tau$,q)')
            plt.legend(loc= 'upper right', fontsize=10)
        #plt.savefig('g2_tau_{t}.png'.format(t= time_step))
        plt.show()
    return g2, lag_steps


def simulate_noise_2d_detector(shape, noise_type='gaussian', mean=0, std_dev=1, lam=10):
    if noise_type == 'gaussian':
        noise = np.random.normal(mean, std_dev, shape)
    elif noise_type == 'poisson':
        noise = np.random.poisson(lam, shape)
    else:
        raise ValueError("Unsupported noise type. Use 'gaussian' or 'poisson'.")
    
    return noise

def animation_generator(frames_ds, step, frame_num):
    print(step)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    #line, = ax.plot([], [], lw=2)
    im=ax.imshow(frames_ds[0],interpolation='none', label = 'Time = 0 sec')
    ax.set_title('Time = 0 s ')
    
    def animate_frames(i):
        im.set_array(frames_ds[i])
        ax.set_title('Time = {:.2f} s (tau = {t})'.format(i*step, t= step ))
        return im
    
    ani = FuncAnimation(fig, animate_frames, repeat=True,
                                        frames=frame_num , interval=50)
    writer = PillowWriter(fps=5,
                                     metadata=dict(artist='Mea'),
                                     bitrate=-1)
    ani.save('zzzframes_{temp}_.gif'.format(temp= step), writer=writer)
    print("done")
#def acf_pacf():


updated_particles =[]
particles_init, radii = initialize_particles(N, Lx, Ly, r_mean , r_std)
updated_particles.append(particles_init)
frame = generate_xray_frames(updated_particles[0], 
                             radii, Lx, Ly, x_pixel_size, 
                             y_pixel_size, x_num_pixel, y_num_pixel)
decay_factors = np.linspace(1,0,int(num_scenes/10))



updated_particles_v2 =[]
updated_particles_v2.append(particles_init)
for k  in range(1, num_scenes):
    u_p = update_particles_v2(updated_particles_v2[k-1],decay_factors[int(k/10)] , D, sample_time_step, Lx, Ly)
    u_p = np.array(u_p)
    updated_particles_v2.append(u_p)  


for i in range(1, num_scenes):
    
    u_p = update_particles(updated_particles[i-1], D, sample_time_step, Lx, Ly)
    u_p = np.array(u_p)
    updated_particles.append(u_p)

# for k in range(int(num_scenes/2)):
#     updated_particles.append(updated_particles[-1])

# len(updated_particles)

q_ring_mask = generate_q_rings_mask((y_num_pixel, x_num_pixel), 1, 50, 10, 3)

g2_all = []
s_all =[]

datasets = []
num_of_datasets = 4
delay_time_step = 0.3
animat_xrax_frames = True


threads = []


animation_tasks_t = []
animation_tasks_n = []

for _i in range(num_of_datasets):
    time_step = time_step_init + (_i*delay_time_step)
    num_frames = int(len(updated_particles_v2)*sample_time_step/time_step)  # Number of frames to simulate
    xray_exposure_scene_step = int(len(updated_particles_v2)/num_frames)
    print(time_step)
    frames_ = Parallel(n_jobs=-1)(delayed(generate_xray_frames)(updated_particles_v2[i], 
                                  radii, Lx, Ly, x_pixel_size, 
                                  y_pixel_size, x_num_pixel, y_num_pixel) for i in tqdm(range(0,len(updated_particles_v2),xray_exposure_scene_step)))

    frames_ = np.array(frames_)
    datasets.append(frames_)
    thread = Thread(target = animation_generator, args = (frames_, time_step, num_frames))
    threads.append(thread)
    thread.start()



for th in range(len(threads)):
    threads[th].join()




async def start_ani():
    results = await asyncio.gather(animation_generator(datasets[0], animation_tasks_t[0], animation_tasks_n[0]), animation_generator(datasets[1], animation_tasks_t[1], animation_tasks_n[1]))
    

loop = asyncio.get_event_loop()
loop.run_until_complete(start_ani())

    
    # if animat_xrax_frames:
    #     fig = plt.figure()
    #     ax = fig.add_subplot(111)
    #     #line, = ax.plot([], [], lw=2)
    #     im=ax.imshow(frames_[0],interpolation='none', label = 'Time = 0 sec')
    #     ax.set_title('Time = 0 s ')
        
    #     def animate_frames(i):
    #         im.set_array(frames_[i])
    #         ax.set_title('Time = {:.2f} s (tau = {t})'.format(i*time_step, t= time_step ))
    #         return im
        
    #     ani = FuncAnimation(fig, animate_frames, repeat=True,
    #                                         frames=num_frames , interval=50)
    #     writer = PillowWriter(fps=5,
    #                                      metadata=dict(artist='Mea'),
    #                                      bitrate=-1)
    #     ani.save('frames_{temp}_.gif'.format(temp= time_step), writer=writer)



intensity_fluctuation_all = []

for _i in range(num_of_datasets):
    
    f = len(datasets[_i])
    f_ = int(f/3*2)
    h = int(f_/100*10)
    
    print(h)
    i_f_ = extract_intensity_for_q_rings(datasets[_i], q_ring_mask)
    intensity_fluctuation_all.append(i_f_)
    slice_point = 50
    for q in range(len(i_f_)):
        
        slice_point = len(i_f_[q])
        
        plt.figure(figsize=(14, 6))

        plt.subplot(2, 1, 1)
        plot_acf(i_f_[q], lags=None,alpha=0.5, ax=plt.gca(), use_vlines=False,)
        plt.xscale('log')
        plt.title('Intensity Fluctuations')
        plt.xlabel('Time')
        plt.ylabel('Intensity')
        
        plt.subplot(2, 1, 2)
        #l = int(slice_point/2)-1
        l = 8
        plot_pacf(i_f_[q][:50], lags=None,alpha=0.5, ax=plt.gca(), method='ywm')
        plt.title('PACF {m} and {n}'.format(m = _i+1, n= q+1))
        plt.xlabel('Lag')
        #plt.xscale('log')
        plt.ylabel('PACF')
        
        plt.tight_layout()
        plt.show()
    #plt.plot(i_f_[0])
    #plt.show() 



for _i in range(num_of_datasets):
    g2_, ts_ = g2_calculation(datasets[_i], q_ring_mask, False)

    plt.plot(ts_, g2_, label= 'DS {z}'.format(z= _i+1))
plt.xscale('log')
plt.xlabel(r'$\tau$ (s)')
plt.ylabel(r'$g^{(2)}$($\tau$,q)')
plt.legend(loc= 'upper right', fontsize=10)
plt.title('step on delay time =  {d}'.format(d=delay_time_step ))
#plt.savefig('g2_tau_step_{t}.png'.format(t= delay_time_step))
plt.show()   
    # if len(frames_)<num_frames_init:
    #    for _ in range(num_frames_init-len(frames_)):
    #        frames_.append(frames_[-1])
    
    # g, s = g2_calculation(frames_, time_step)
    # s_all.append(s)
    # g2_all.append(g)
#particles = np.array(particles)

q_ring_num = 5

for j in range(len(g2_all)):
    plt.plot(s_all[j],g2_all[j][:,q_ring_num], label='Lag time: {t}'.format(t = j))
    plt.xscale('log')
    plt.xlabel(r'$\tau$ (s)')
    plt.ylabel(r'$g^{(2)}$($\tau$,q)')
    plt.legend(loc= 'upper right', fontsize=10)
#plt.label('Q_ring number: {n}'.format(n=q_ring_num))
plt.savefig('g2_tau_{t}.png'.format(t= time_step))
plt.show()   




num_of_animation_scence = int(num_scenes * sample_time_step /scene_animation_time_step)
animation_step = int(num_scenes/num_of_animation_scence)


#updated_particles = np.array(updated_particles)




fig, ax = plt.subplots()
ax.set_xlim(0, Lx)
ax.set_ylim(0, Ly)
ax.set_xticks([])
ax.set_yticks([])
time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
scat = ax.scatter(particles_init[:, 0], particles_init[:, 1], color = cm.rainbow(np.linspace(0, 1, N)), s=np.pi * radii**2)


def animate(i):
    scat.set_offsets(updated_particles[(i*animation_step)-1])
    time_text.set_text('Time = {:.2f} s'.format(i* animation_step * sample_time_step))
    rect = Rectangle((int(Lx/2)-int(x_beam_size/2), int(Ly/2)-int(y_beam_size/2)), x_beam_size, y_beam_size, linewidth=1, edgecolor='r', facecolor='none')
    ax.add_patch(rect)
    return scat,


def animate_v2(i):
    scat.set_offsets(updated_particles_v2[(i*animation_step)-1])
    time_text.set_text('Time = {:.2f} s'.format(i* animation_step * sample_time_step))
    rect = Rectangle((int(Lx/2)-int(x_beam_size/2), int(Ly/2)-int(y_beam_size/2)), x_beam_size, y_beam_size, linewidth=1, edgecolor='r', facecolor='none')
    ax.add_patch(rect)
    return scat,


ani = FuncAnimation(fig, animate, repeat=True,
                                    frames=num_of_animation_scence , interval=50)

# To save the animation using Pillow as a gif
writer = PillowWriter(fps=5*xray_exposure_scene_step,
                                 metadata=dict(artist='Mea'),
                                 bitrate=-1)
ani.save('scatter_a{temp}.gif'.format(temp= T), writer=writer)

fig, ax = plt.subplots()
ax.set_xlim(0, Lx)
ax.set_ylim(0, Ly)
ax.set_xticks([])
ax.set_yticks([])
time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
scat = ax.scatter(particles_init[:, 0], particles_init[:, 1], color = cm.rainbow(np.linspace(0, 1, N)), s=np.pi * radii**2)


ani_v2 = FuncAnimation(fig, animate_v2, repeat=True,
                                    frames=num_of_animation_scence , interval=50)

# To save the animation using Pillow as a gif
writer = PillowWriter(fps=5*xray_exposure_scene_step,
                                 metadata=dict(artist='Mea'),
                                 bitrate=-1)
ani_v2.save('scatter_v2{temp}.gif'.format(temp= T), writer=writer)






#np.max(g2)
"""

s = 200

x = np.linspace(-1, 1, s)
y = np.linspace(-1, 1, s)
xx, yy = np.meshgrid(x, y)  

  
_r = r*1e10
particle_area = np.pi * _r**2

electron_density = np.zeros_like(xx)

for particle in updated_particles[10]:
    r_squared = (xx - (particle[0]))**2 + (yy - (particle[1]))**2
    mask = r_squared <= _r**2
    electron_density += mask 
    
fft_result = np.fft.fft2(electron_density)
fft_result_shifted = np.fft.fftshift(fft_result)

# Calculate magnitude of the FFT result (spectrum)
saxs_spectrum = np.abs(fft_result_shifted)

plt.imshow(saxs_spectrum)
plt.show()





import numpy as np
import matplotlib.pyplot as plt

# Constants
num_particles = 100
num_frames = 100
sample_time_step = 0.01
diffusion_coefficient = 0.1
Lx, Ly = 10, 10  # Dimensions of the sample area
detector_size = (500, 500)  # Size of the 2D detector

# Initialize particle positions
positions = np.random.rand(num_particles, 2) * np.array([Lx, Ly])

# Function to update particle positions using Brownian motion
def update_positions(positions, D, dt, Lx, Ly):
    displacement = np.sqrt(2 * D * dt) * np.random.randn(*positions.shape)
    positions += displacement
    # Ensure periodic boundary conditions
    positions = positions % np.array([Lx, Ly])
    return positions

# Function to compute the SAXS pattern
def compute_saxs_pattern(positions, detector_size, Lx, Ly):
    qx = np.fft.fftfreq(detector_size[0], d=Lx/detector_size[0])
    qy = np.fft.fftfreq(detector_size[1], d=Ly/detector_size[1])
    qx, qy = np.meshgrid(qx, qy)
    q_squared = qx**2 + qy**2

    # Scattering intensity
    intensity = np.zeros(detector_size)
    for pos in positions:
        phase = np.exp(-1j * (qx * pos[0] + qy * pos[1]))
        intensity += np.abs(np.fft.fftshift(np.fft.fft2(phase)))**2

    return intensity / num_particles

# Initialize the intensity pattern
intensity_pattern = np.zeros(detector_size)

# Simulate Brownian motion and compute SAXS patterns
for frame in range(num_frames):
    positions = update_positions(positions, diffusion_coefficient, sample_time_step, Lx, Ly)
    intensity_pattern += compute_saxs_pattern(positions, detector_size, Lx, Ly)

# Average the intensity pattern over all frames
intensity_pattern /= num_frames

# Plot the resulting SAXS pattern
plt.imshow(np.log(intensity_pattern + 1), cmap='viridis', extent=(-Lx/2, Lx/2, -Ly/2, Ly/2))
plt.colorbar(label='Log Intensity')
plt.title('Simulated SAXS Pattern')
plt.xlabel('qx')
plt.ylabel('qy')
plt.show()



import numpy as np
import matplotlib.pyplot as plt

def generate_sample(num_particles, Lx, Ly):
    x = np.random.uniform(-Lx/2, Lx/2, num_particles)
    y = np.random.uniform(-Ly/2, Ly/2, num_particles)
    return np.vstack((x, y)).T

def brownian_motion(particles, dt, diffusion_coefficient):
    displacement = np.sqrt(2 * diffusion_coefficient * dt) * np.random.randn(*particles.shape)
    particles += displacement
    return particles

def form_factor(q, radius):
    # Form factor for spherical particles
    qr = q * radius
    return (3 * (np.sin(qr) - qr * np.cos(qr)) / qr**3)**2

def calculate_scattering(qx, qy, particle_positions, radius):
    intensity = np.zeros(qx.shape)
    q = np.sqrt(qx**2 + qy**2)
    Fq = form_factor(q, radius)
    for pos in particle_positions:
        phase = np.exp(1j * (qx * pos[0] + qy * pos[1]))
        intensity += Fq * np.abs(phase)**2
    return intensity

def simulate_xpcs(num_particles, num_steps, dt, diffusion_coefficient, Lx, Ly, nx, ny, q_range, radius):
    sample = generate_sample(num_particles, Lx, Ly)
    qx = np.linspace(-q_range, q_range, nx)
    qy = np.linspace(-q_range, q_range, ny)
    qx, qy = np.meshgrid(qx, qy)
    
    intensity = np.zeros((num_steps, nx, ny))
    
    for step in range(num_steps):
        sample = brownian_motion(sample, dt, diffusion_coefficient)
        intensity[step] = calculate_scattering(qx, qy, sample, radius)
    
    return intensity

def plot_speckle_pattern(intensity):
    plt.figure(figsize=(8, 6))
    plt.imshow(intensity, extent=[-1, 1, -1, 1], origin='lower', cmap='viridis')
    plt.colorbar(label='Intensity')
    plt.xlabel('qx')
    plt.ylabel('qy')
    plt.title('XPCS Speckle Pattern')
    plt.show()

# Simulation parameters
num_particles = 10  # Number of particles in the sample
num_steps = 10  # Number of time steps
dt = 0.1  # Time step
diffusion_coefficient = 1e-3  # Diffusion coefficient
Lx, Ly = 100, 100  # Sample dimensions
nx, ny = 500, 500  # Detector resolution
q_range = 0.1  # Range of q values
radius = 1.0  # Particle radius

# Simulate and plot the XPCS speckle pattern
intensity = simulate_xpcs(num_particles, num_steps, dt, diffusion_coefficient, Lx, Ly, nx, ny, q_range, radius)

# Plot the speckle pattern at the last time step
plot_speckle_pattern(intensity[0])

"""
