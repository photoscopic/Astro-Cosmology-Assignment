#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 24 02:14:27 2023

@author: ameyaparab
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad


H0 = 70.0 
redshifts = np.array([0, 2, 6, 1100])
def hubble_parameter(z, O_m, O_DE, w_DE):
    return H0 * np.sqrt(O_m * (1 + z)**3 + O_DE * np.exp(3 * quad(lambda z_prime: (1 + w_DE) / (1 + z_prime), 0, z)[0]))


def age_of_universe(z_values, O_m, O_DE, w_DE):
    age_of_universe_values = []
    for z in z_values:
        integrand = lambda z_prime: 1 / ((1 + z_prime) * hubble_parameter(z_prime, O_m, O_DE, w_DE))
        result, _ = quad(integrand, z, np.inf)
        age_of_universe_values.append(result)
    return age_of_universe_values


O_m_i = 0.3
O_DE_i = 0.7
w_DE_i = -1
age_of_universe_lcdm = np.array(age_of_universe(redshifts, O_m_i, O_DE_i, w_DE_i)) * 1e3  # convert to Gyr


O_m_ii = 0.3
O_DE_ii = 0.7
w_DE_ii = -0.7
age_of_universe_wcdm1 = np.array(age_of_universe(redshifts, O_m_ii, O_DE_ii, w_DE_ii)) * 1e3


O_m_iii = 0.3
O_DE_iii = 0.7
w_DE_iii = -1.2
age_of_universe_wcdm2 = np.array(age_of_universe(redshifts, O_m_iii, O_DE_iii, w_DE_iii)) * 1e3


O_m_iv = 0.8
O_DE_iv = 0.2
w_DE_iv = -1
age_of_universe_lcdm_omega_m = np.array(age_of_universe(redshifts, O_m_iv, O_DE_iv, w_DE_iv)) * 1e3


O_m_v = 0.3
O_DE_v = 0.7
w_DE_v = -1
age_of_universe_lcdm_T0 = np.array(age_of_universe(redshifts, O_m_v, O_DE_v, w_DE_v)) * 1e3


plt.figure(figsize=(10, 6))
plt.plot(redshifts, age_of_universe_lcdm, label='Flat LCDM')
plt.plot(redshifts, age_of_universe_wcdm1, label='Flat wCDM, w = -0.7')
plt.plot(redshifts, age_of_universe_wcdm2, label='Flat wCDM, w = -1.2')
plt.plot(redshifts, age_of_universe_lcdm_omega_m, label='Flat LCDM, Omega_m = 0.8')
plt.plot(redshifts, age_of_universe_lcdm_T0, label='Flat LCDM, T0 = 100 K')

# plt.semilogy(redshifts, age_of_universe_lcdm, label='Flat LCDM')
# plt.semilogy(redshifts, age_of_universe_wcdm1, label='Flat wCDM, w = -0.7')
# plt.semilogy(redshifts, age_of_universe_wcdm2, label='Flat wCDM, w = -1.2')
# plt.semilogy(redshifts, age_of_universe_lcdm_omega_m, label='Flat LCDM, Omega_m = 0.8')
# plt.semilogy(redshifts, age_of_universe_lcdm_T0, label='Flat LCDM, T0 = 100 K')
plt.title('Age of the Universe for Different Models')
plt.xlabel('Redshift (z)')
plt.ylabel('Age of the Universe (Gyr)')
plt.legend()
plt.grid(True)
plt.show()
