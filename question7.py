#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 22 22:42:19 2023

@author: ameyaparab
"""

import numpy as np
import matplotlib.pyplot as plt


h = 6.62607015e-34   
c = 3*10**5 
k = 1.380649e-23  

T0_CMB = 2.725

redshifts = [0, 6, 20, 1090, 10**4]


frequencies = np.logspace(6, 15, 1000) 


for z in redshifts:
    T = T0_CMB * (1 + z)
    spectral_radiance = (8 * np.pi * frequencies**2 * h) / (c**3) / (np.exp((h * frequencies) / (k * T)) - 1)
    plt.plot(frequencies, spectral_radiance, label=f'z = {z}')

plt.title('Blackbody Radiation Spectrum of the Universe')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Spectral Radiance (W m^-2 Hz^-1 sr^-1)')
plt.legend()
plt.show()
