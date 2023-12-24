import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


redshifts = np.array([0.01, 0.012, 0.016, 0.02167, 0.0343, 0.0593]) *3e5
luminosity_distances = np.array([38.8, 60.6, 73.8, 87.0, 140.4, 268.7])


def fit(x, m, c):
    return m * x - c

params, covariance = curve_fit(fit, redshifts, luminosity_distances)


m, c = params

plt.scatter(redshifts, luminosity_distances, label='Data')
plt.plot(redshifts,fit(redshifts, m, c), color='red', label=f'Fit: y = {1/m:.2f}x + {c:.2f}')
plt.xlabel('Redshift (cz)')
plt.ylabel('Luminosity Distance (Mpc)')
plt.legend()
plt.show()

# Display the fitted parameters
print(f'Fitted slope (m): {1/m:.4f}')
print(f'Fitted intercept (c): {c:.4f}')
