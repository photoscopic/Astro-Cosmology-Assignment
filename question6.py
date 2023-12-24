#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 22 22:15:42 2023

@author: ameyaparab
"""

import numpy as np

# Constants
G = 6.67430e-11 
c = 299792458    
H0 = 70 


T0 = 2.725        

def hubble_parameter(z, w):
    rho_total = rho_matter*(1 + z)**3 + rho_radiation*(1 + z)**4 + rho_lambda + rho_extra*(1 + z)**(3*(1+w))
    p_total = p_matter*(1 + z)**3 + p_radiation*(1 + z)**4 + p_lambda + p_extra*(1 + z)**(3*(1+w))

    hz = (8 * np.pi * G)/(3 * c**2) * (rho_total - w * p_total)
    
    return np.sqrt(hz)


redshift_values = [0, 6, 20, 1090, 10**4]

w_values = [2/3, 1]

rho_matter = 0.30*3 * H0**2/(8 * np.pi * G)
rho_radiation = 1e-4
rho_lambda = 0.7*3 * H0**2/(8 * np.pi * G)

p_matter = 0
p_radiation = rho_radiation/3
p_lambda = -rho_lambda


w_extra = 1 
rho_extra = 0.01*rho_matter  
p_extra = w_extra*rho_extra


for w in w_values:
    for z in redshift_values:
        hubble = hubble_parameter(z, w)
        print(f"z = {z}, w = {w}: H(z) = {hubble} km/s/Mpc")
