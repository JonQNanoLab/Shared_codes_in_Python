import numpy as np
import time as time
import multiprocessing as mp
import matplotlib.pyplot as plt
from MainTools import EHinc_array, C_Ext_array, AGmatrix_array
from FiniteArrayTools import lambda_kerker, ab_MieCoef


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
N = 50

# Radius of the nanoparticles
a = 150*nm

# Definition of the incident EM field
Kerker = True
cf1 = 1         # Polarization coefficient in the U1 direction (not normalized)
cf2 = +1j       # Polarization coefficient in the U1 direction (not normalized)

norm = np.sqrt(cf1*np.conj(cf1) + cf2*np.conj(cf2))
ncf1 = cf1/norm  # Polarization coefficient in the U1 direction (normalized)
ncf2 = cf2/norm  # Polarization coefficient in the U2 direction (normalized)


if not Kerker:
    # Choose a wavelength number from imported data
    nwave = 525
    Lambda = np.array([LambdaArray[nwave]])
    k = 2*np.pi/Lambda

    # Mie coefficients for the chosen wavelength
    a_1, b_1 = ab_MieCoef(a, Lambda, np.array([eps_part[nwave]]),
                          np.array([1.0]), 1)

    # Electric and magnetic polarizabilities
    a_e = 1j*(6*np.pi/k**3)*a_1
    a_m = 1j*(6*np.pi/k**3)*b_1

else:
    Lambda, a_e, a_m = lambda_kerker(a, LambdaArray, eps_part,
                                     np.ones(len(eps_part)), 1)
    k = 2*np.pi/Lambda


# ---------- OPTIONS OF THE PROGRAM ----------------------------------------- #
# --------------------------------------------------------------------------- #

thetalist = np.linspace(np.pi/2, np.pi/2, 1)
Dplist = np.linspace(2*a, 700*nm, 50)
Cext_plot = np.zeros((len(Dplist), len(thetalist)))

t1 = time.time()
for kd in range(0, len(Dplist)):
    Dp = Dplist[kd]
    print(kd)

    # For each value of the Dp value, we calculate the dipole coupling
    M = AGmatrix_array(Dp, N, a_e, a_m, k)

    def func(theta):
        """Parallelizing piece of code."""
        # Calculate the EM field vector in each of the dipole sites
        EHvec = EHinc_array(N, Dp, theta, [ncf1, ncf2], k, 'PW')

        # Calculate the self-consistent dipole coupling
        P = M.dot(EHvec)

        C_ext = C_Ext_array(N, Dp, P, theta, [ncf1, ncf2], k)

        return float(C_ext)

    # Generate processes equal to the number of cores
    ncore = 8
    pool = mp.Pool(processes=ncore)

    # Distribute the parameter sets evenly across the cores
    for i, res in enumerate(pool.imap(func, thetalist), 0):
        Cext_plot[kd][i] = res

    pool.close()

if len(thetalist) != 1 and len(Dplist) != 1:
    # Figure parameters

    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)      # subplot integer, three digit number
    ax1.set_xlabel('Incidence angle (rad)')
    ax1.set_ylabel('D´ (nm)')
    ax1.set_title('Extinction cross section (nm2)')
    thetaplot, Dpplot = np.meshgrid(thetalist, Dplist)
    ax1.contourf(thetaplot, Dpplot/1e-9, Cext_plot, 1000)

if len(thetalist) != 1 and len(Dplist) == 1:
    # Figure parameters

    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)      # subplot integer, three digit number
    ax1.set_xlabel('Incidence angle (rad)')
    ax1.set_ylabel('Extinction cross section (nm2)')
    plt.plot(thetalist, np.transpose(Cext_plot))

if len(thetalist) == 1 and len(Dplist) != 1:
    # Figure parameters

    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)      # subplot integer, three digit number
    ax1.set_xlabel('D´ (nm)')
    ax1.set_ylabel('Extinction cross section (nm2)')
    plt.plot(Dplist/nm, Cext_plot)

plt.show()
