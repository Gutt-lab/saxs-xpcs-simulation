#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 08:56:03 2024

@author: tosson
"""

class SampleEnviroment:
    
    sample_enviroment_dict = {
        #"box_length": 0 
        # "box_width":0, 
        # "number_of_particles":0, 
        # "mean_particle_radius":0.0, 
        # "std_particle_radius":0.0,
        # "temprature_in_kelvin":0.0,
        # "viscosity_of_buffer":0.0,
        # "dynamics_time":0.0,
        }
    
    def __intialize_sample_enviroment(self, *args, **kwargs ):
        print(kwargs )
        
    def setaa(self):
        self.__intialize_sample_enviroment(t = 2)
    
    # Lx = 1000  # Box length in x-direction (um)
    # Ly = 1000  # Box length in y-direction (um)
    # N  = 200  # Number of particles
    # r_mean = 10.00  # Mean particle radius (um)
    # r_std  = 1.5   # Standard deviation of particle radius (um)
    # k_B = 1.380649e-23  # Boltzmann constant in J/K
    # T = 273  # Temperature in Kelvin
    # eta = 19e-3 # Viscosity of buffer in Pa.s
    # r = 1e-9  # Radius of the particle in meters
    # gamma = 6 * np.pi * eta * r  # Drag coefficient
    # D = k_B * T / gamma  # Diffusion coefficient
    # D = D*1e12   
    # #D_0 = 6.0e-10      # Diffusion coefficient [m^2/s]
    # #D = D_0*1e12       # Diffusion coefficient [um^2/s]
    
    # lab_time = 100.0 # [s]
    # sample_time_step = 0.1 # The dynamic time [s]
    # scene_animation_time_step = 0.1
    # num_scenes = int(lab_time/sample_time_step)  
    

s = SampleEnviroment()
s.sample_enviroment_dict["box_length"] = 10
s.setaa()
