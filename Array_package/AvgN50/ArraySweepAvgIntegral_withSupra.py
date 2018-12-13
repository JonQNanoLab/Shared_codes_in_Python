import numpy as np
import multiprocessing as mp
import scipy.io as sio  # To read .mat files
from FiniteArrayTools import lambda_kerker, ab_MieCoef
from AvgIntegral import AvgIntegral


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

# Radius for which we calculate the avg. integral of fCD
R = a + 1*nm


# Initializing arrays and matrices
thetalist = np.linspace(0, np.pi/2, 100)
Dplist = np.linspace(2*a, 3000*nm, 601)
fCDIntegral = np.zeros((len(Dplist), len(thetalist)), dtype=float)

Psupra = sio.loadmat('Psupra.mat')['Psupra']

# Once pool has been closed we proceed to calculate the average integral.
# This allows for reutilizing the pool of workers again to parallelize the
# calculus of the average integral.

# loop to be parallelized
for kd in range(0, len(Dplist)):
    print("Calculating fCD average for kd = " + str(kd))
    print(kd)

    def func2(theta):
        """Parallelizing piece of code 2."""
        ktheta = list(thetalist).index(theta)

        P = Psupra[kd][ktheta][:]
        Dp = Dplist[kd]
        theta = thetalist[ktheta]

        fCDInt = 1/N*AvgIntegral(P, Dp, theta, R)

        return fCDInt

    # Generate processes equal to the number of cores
    ncore = 8
    pool = mp.Pool(processes=ncore)

    print("POOL HAS BEEN OPENED!!")
    # Distribute the parameter sets evenly across the cores
    for ktheta, res in enumerate(pool.imap(func2, thetalist), 0):
        fCDIntegral[kd][ktheta] = res

    pool.close()
print("POOL HAS BEEN CLOSED!!")

sio.savemat('fCDIntegral_total.mat', {'fCDSweepN'+str(N): fCDIntegral})
