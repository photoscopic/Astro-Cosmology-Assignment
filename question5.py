#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 24 01:58:37 2023

@author: ameyaparab
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.constants import c

# Constants
H0 = 70.0  
c = 3e5   


def hubble_integrand(z, O_m, O_DE, w_DE):
    return 1 / np.sqrt(O_m * (1 + z)**3 + O_DE * np.exp(3 * quad(lambda z_prime: (1 + w_DE) / (1 + z_prime), 0, z)[0]))

def luminosity_distance(z, O_m, O_DE, w_DE):
    integrand, _ = quad(lambda z_prime: hubble_integrand(z_prime, O_m, O_DE, w_DE), 0, z)
    return (1 + z) * c / H0 * integrand


def comoving_distance(z, O_m, O_DE, w_DE):
    integrand, _ = quad(lambda z_prime: 1 / hubble_integrand(z_prime, O_m, O_DE, w_DE), 0, z)
    return c / H0 * integrand


def angular_diameter_distance(z, O_m, O_DE, w_DE):
    return comoving_distance(z, O_m, O_DE, w_DE) / (1 + z)


def proper_distance(z, O_m, O_DE, w_DE):
    return comoving_distance(z, O_m, O_DE, w_DE) / (1 + z)


z_values = np.linspace(0, 10, 100)


plt.figure(figsize=(12, 8))




O_m_i = 0.3
O_DE_i = 0.7
w_DE_i = -1
d_L_i = [luminosity_distance(z, O_m_i, O_DE_i, w_DE_i) for z in z_values]
d_C_i = [comoving_distance(z, O_m_i, O_DE_i, w_DE_i) for z in z_values]
d_A_i = [angular_diameter_distance(z, O_m_i, O_DE_i, w_DE_i) for z in z_values]
d_P_i = [proper_distance(z, O_m_i, O_DE_i, w_DE_i) for z in z_values]


O_m_ii = 0.3
O_DE_ii = 0.7
w_DE_ii = -0.7
d_L_ii = [luminosity_distance(z, O_m_ii, O_DE_ii, w_DE_ii) for z in z_values]
d_C_ii = [comoving_distance(z, O_m_ii, O_DE_ii, w_DE_ii) for z in z_values]
d_A_ii = [angular_diameter_distance(z, O_m_ii, O_DE_ii, w_DE_ii) for z in z_values]
d_P_ii = [proper_distance(z, O_m_ii, O_DE_ii, w_DE_ii) for z in z_values]


O_m_iii = 0.3
O_DE_iii = 0.7
w_DE_iii = -1.2
d_L_iii = [luminosity_distance(z, O_m_iii, O_DE_iii, w_DE_iii) for z in z_values]
d_C_iii = [comoving_distance(z, O_m_iii, O_DE_iii, w_DE_iii) for z in z_values]
d_A_iii = [angular_diameter_distance(z, O_m_iii, O_DE_iii, w_DE_iii) for z in z_values]
d_P_iii = [proper_distance(z, O_m_iii, O_DE_iii, w_DE_iii) for z in z_values]


O_m_iv = 0.8
O_DE_iv = 0.2
w_DE_iv = -1
d_L_iv = [luminosity_distance(z, O_m_iv, O_DE_iv, w_DE_iv) for z in z_values]
d_C_iv = [comoving_distance(z, O_m_iv, O_DE_iv, w_DE_iv) for z in z_values]
d_A_iv = [angular_diameter_distance(z, O_m_iv, O_DE_iv, w_DE_iv) for z in z_values]
d_P_iv = [proper_distance(z, O_m_iv, O_DE_iv, w_DE_iv) for z in z_values]


O_m_v = 0.3
O_DE_v = 0.7
w_DE_v = -1
d_L_v = [luminosity_distance(z, O_m_v, O_DE_v, w_DE_v) for z in z_values]
d_C_v = [comoving_distance(z, O_m_v, O_DE_v, w_DE_v) for z in z_values]
d_A_v = [angular_diameter_distance(z, O_m_v, O_DE_v, w_DE_v) for z in z_values]
d_P_v = [proper_distance(z, O_m_v, O_DE_v, w_DE_v) for z in z_values]

plt.subplot(2, 2, 1)
plt.plot(z_values, d_L_i, label='Model (i)')
plt.plot(z_values, d_L_ii, label='Model (ii)')
plt.plot(z_values, d_L_iii, label='Model (iii)')
plt.plot(z_values, d_L_iv, label='Model (iv)')
plt.plot(z_values, d_L_v, label='Model (v)')
plt.title('Luminosity Distance')
plt.xlabel('Redshift')
plt.ylabel('Distance (Mpc)')
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(z_values, d_C_i, label='Model (i)')
plt.plot(z_values, d_C_ii, label='Model (ii)')
plt.plot(z_values, d_C_iii, label='Model (iii)')
plt.plot(z_values, d_C_iv, label='Model (iv)')
plt.plot(z_values, d_C_v, label='Model (v)')
plt.title('Comoving Distance')
plt.xlabel('Redshift')
plt.ylabel('Distance (Mpc)')
plt.legend()

plt.subplot(2, 2, 3)
plt.plot(z_values, d_A_i, label='Model (i)')
plt.plot(z_values, d_A_ii, label='Model (ii)')
plt.plot(z_values, d_A_iii, label='Model (iii)')
plt.plot(z_values, d_A_iv, label='Model (iv)')
plt.plot(z_values, d_A_v, label='Model (v)')
plt.title('Angular Diameter Distance')
plt.xlabel('Redshift')
plt.ylabel('Distance (Mpc)')
plt.legend()

plt.subplot(2, 2, 4)
plt.plot(z_values, d_P_i, label='Model (i)')
plt.plot(z_values, d_P_ii, label='Model (ii)')
plt.plot(z_values, d_P_iii, label='Model (iii)')
plt.plot(z_values, d_P_iv, label='Model (iv)')
plt.plot(z_values, d_P_v, label='Model (v)')
plt.title('Proper Distance')
plt.xlabel('Redshift')
plt.ylabel('Distance (Mpc)')
plt.legend()

plt.tight_layout()
plt.show()
