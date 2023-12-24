import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad


H0 = 70.0 
redshifts = np.array([0, 2, 6, 1100])

def hubble_parameter(z, O_m, O_DE, w_DE):
    return H0 * np.sqrt(O_m * (1 + z)**3 + O_DE * np.exp(3 * quad(lambda z_prime: (1 + w_DE) / (1 + z_prime), 0, z)[0]))


def lookback_time(z_values, O_m, O_DE, w_DE):
    lookback_times = []
    for z in z_values:
        integrand = lambda z_prime: 1 / ((1 + z_prime) * hubble_parameter(z_prime, O_m, O_DE, w_DE))
        result, _ = quad(integrand, 0, z)
        lookback_times.append(result)
    return lookback_times


O_m_i = 0.3
O_DE_i = 0.7
w_DE_i = -1
lookback_time_lcdm = np.array(lookback_time(redshifts, O_m_i, O_DE_i, w_DE_i))*3e5


O_m_ii = 0.3
O_DE_ii = 0.7
w_DE_ii = -0.7
lookback_time_wcdm1 =  np.array(lookback_time(redshifts, O_m_ii, O_DE_ii, w_DE_ii))*3e5


O_m_iii = 0.3
O_DE_iii = 0.7
w_DE_iii = -1.2
lookback_time_wcdm2 = np.array(lookback_time(redshifts, O_m_iii, O_DE_iii, w_DE_iii))*3e5


O_m_iv = 0.8
O_DE_iv = 0.2
w_DE_iv = -1
lookback_time_lcdm_omega_m = np.array(lookback_time(redshifts, O_m_iv, O_DE_iv, w_DE_iv))*3e5


O_m_v = 0.3
O_DE_v = 0.7
w_DE_v = -1
lookback_time_lcdm_T0 = np.array(lookback_time(redshifts, O_m_v, O_DE_v, w_DE_v))*3e5


plt.figure(figsize=(10, 6))
plt.plot(redshifts, lookback_time_lcdm, label='Flat LCDM')
plt.plot(redshifts, lookback_time_wcdm1, label='Flat wCDM, w = -0.7')
plt.plot(redshifts, lookback_time_wcdm2, label='Flat wCDM, w = -1.2')
plt.plot(redshifts, lookback_time_lcdm_omega_m, label='Flat LCDM, Omega_m = 0.8')
plt.plot(redshifts, lookback_time_lcdm_T0, label='Flat LCDM, T0 = 100 K')

plt.title('Lookback Time of the Universe for Different Models')
plt.xlabel('Redshift (z)')
plt.ylabel('Lookback Time (Gyr)')
plt.legend()
plt.grid(True)
plt.show()
