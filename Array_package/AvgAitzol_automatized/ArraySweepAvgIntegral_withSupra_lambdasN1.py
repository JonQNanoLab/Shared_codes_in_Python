import numpy as np
# import multiprocessing as mp
import matplotlib.pyplot as plt
import scipy.io as sio  # To read .mat files
from MainTools import EHinc_array, AGmatrix_array
from FiniteArrayTools import ab_MieCoef
from Avgintegral_run_function import Avgintegral_run_function


# ---------- CONSTANTS ------------------------------------------------------ #
# --------------------------------------------------------------------------- #

nm = 10**(-9)
eps_0 = 8.854187817*10**(-12)
mu_0 = 4*np.pi*10**(-7)
Z_0 = np.sqrt(mu_0/eps_0)
clight = 1/np.sqrt(eps_0*mu_0)


# ---------- PARTICLE PERMITTIVITY ------------------------------------------ #
# --------------------------------------------------------------------------- #

# Read the data of the elec. permittivity of the particles embedded in vacuum.
# This file needs to be contained in the same folder as this program. The
# aforementioned file must contain the wavelength in the first column, the
# real part of the refractive index in the second column and the imaginary
# part of the refractive index in the third column.

file = open('Si_data.txt', 'r')
lines = file.readlines()
la_read = []
ne_read = []
ke_read = []

for x in lines:
    la_read.append(x.split(' ')[0])
    ne_read.append(x.split(' ')[1])
    ke_read.append(x.split(' ')[2])

file.close()

LambdaArray = np.array([float(i) for i in la_read])
n_eps = np.array([float(i) for i in ne_read])
k_eps = np.array([float(i) for i in ke_read])
eps_part = (n_eps + 1j*k_eps)**2  # Check the sign of complex sum!


# ---------- MAIN PARAMETERS OF THE PROBLEM --------------------------------- #
# --------------------------------------------------------------------------- #

# Number of nanoparticles
N = 1

# Radius of the nanoparticles
a = 75*nm

# Definition of the incident EM field
cf1 = 1         # Polarization coefficient in the U1 direction (not normalized)
cf2 = +1j       # Polarization coefficient in the U1 direction (not normalized)

norm = np.sqrt(cf1*np.conj(cf1) + cf2*np.conj(cf2))
ncf1 = cf1/norm  # Polarization coefficient in the U1 direction (normalized)
ncf2 = cf2/norm  # Polarization coefficient in the U2 direction (normalized)


# ---------- OPTIONS OF THE PROGRAM ----------------------------------------- #
# --------------------------------------------------------------------------- #

# Input parameters
Dp = 1208*nm
theta_i = 0
R = a + 1*nm  # Radius at which average is being calculated

fCDsum_tot = np.zeros(len(LambdaArray))
# Calculate for lambda in [400nm, 800nm]
for kwave in range(300, 701):
    print(kwave)

    # We need to calculate the parameters that depend on Lambda: k, a_e, a_m
    nwave = kwave
    Lambda = np.array([LambdaArray[nwave]])
    k = 2*np.pi/Lambda

    # Mie coefficients for the chosen wavelength
    a_1, b_1 = ab_MieCoef(a, Lambda, np.array([eps_part[nwave]]),
                          np.array([1.0]), 1)

    # Electric and magnetic polarizabilities
    a_e = 1j*(6*np.pi/k**3)*a_1
    a_m = 1j*(6*np.pi/k**3)*b_1

    # Calculus of the dipole distribution
    EHvec = EHinc_array(N, Dp, theta_i, [ncf1, ncf2], k, 'PW')
    M = AGmatrix_array(Dp, N, a_e, a_m, k)
    P = M.dot(EHvec)

    # Calculus of the average fCD integral
    EHinc_param = [theta_i, [ncf1, ncf2], k]
    fCDsum_tot[kwave] = Avgintegral_run_function(P, Dp, R, EHinc_param)


sio.savemat('fCDavg.mat', {'fCDavg': fCDsum_tot/N})

plt.plot(LambdaArray/nm, fCDsum_tot, '.')
plt.xlabel('Wavelength (nm)')
plt.ylabel('fCD_avg')
plt.title('Average CD enhancement')
plt.xlim((400, 800))
plt.show()
