"""
Simulacija sistema masa-opruga-prigušivač.

Zajedničke konstante, jednačina i parametri simulacije
koje koriste animacija.py i Jupyter notebook.
"""

import numpy as np
from scipy.integrate import odeint

# === Konstante sistema ===
M = 3.2e-2  # kg - masa
c = 3e5     # N/m - konstanta opruge
A = 5e-5    # m^2 - površina
P = 40e5    # N/m^2 - pritisak
u = 1       # koeficijent prigušenja


def jednacina(y, t):
    """Sistem diferencijalnih jednačina prvog reda."""
    x, v = y
    dydt = [v, (A * P * np.sin(2 * np.pi * t) - u * v - c * x) / M]
    return dydt


# === Parametri simulacije ===
y0 = [0, 0]
T_pocetak = 0
T_kraj = 3
f_odabiranja = 10000
T_odabiranja = 1 / f_odabiranja
t = np.arange(T_pocetak, T_kraj, T_odabiranja)


def resi():
    """Rešava diferencijalnu jednačinu i vraća rezultat."""
    return odeint(jednacina, y0, t)
