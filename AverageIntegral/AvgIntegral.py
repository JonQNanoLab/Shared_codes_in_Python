import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from MainTools import EHinc_array, AGmatrix_array
from GreenFunc import Ge, Gm
from FiniteArrayTools import lambda_kerker, ab_MieCoef, EH0


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


# ---------- DIPOLE POSITION DATA ORIGIN ------------------------------------ #
# --------------------------------------------------------------------------- #

# The program allows for reading the spatial distribution of dipoles either
# from a file 'DipolePos.txt' or directly using functions for a given geometry.
# Variable Fromfile determines which of both ways we are using.

Fromfile = False


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


theta = np.pi/2
Dp = 620*nm

# For the Dp value, we calculate the dipole coupling
M = AGmatrix_array(Dp, N, a_e, a_m, k)

# Calculate the EM field vector in each of the dipole sites
EHvec = EHinc_array(N, Dp, theta, [ncf1, ncf2], k, 'PW')

# Calculate the self-consistent dipole coupling
P = M.dot(EHvec)

# Calculate the absolute value of the dipoles in each site
Pabs = np.zeros(N, dtype=float)
Mabs = np.zeros(N, dtype=float)
Px = np.zeros(N, dtype=float)
Py = np.zeros(N, dtype=float)
Pz = np.zeros(N, dtype=float)
Mx = np.zeros(N, dtype=float)
My = np.zeros(N, dtype=float)
Mz = np.zeros(N, dtype=float)
for i in range(0, N):

    px = P[3*i]
    py = P[3*i + 1]
    pz = P[3*i + 2]

    Pabs[i] = np.sqrt(px*np.conj(px) + py*np.conj(py) + pz*np.conj(pz))

    mx = P[3*i + 3*N]
    my = P[3*i + 3*N + 1]
    mz = P[3*i + 3*N + 2]

    Mabs[i] = np.sqrt(mx*np.conj(mx) + my*np.conj(my) + mz*np.conj(mz))

    Px[i] = np.abs(px)*np.cos(np.angle(px))
    Py[i] = np.abs(py)*np.cos(np.angle(py))
    Pz[i] = np.abs(pz)*np.cos(np.angle(pz))

    Mx[i] = np.abs(mx)*np.cos(np.angle(mx))
    My[i] = np.abs(my)*np.cos(np.angle(my))
    Mz[i] = np.abs(mz)*np.cos(np.angle(mz))


NPmax = np.argmax(Pabs)

# Plot the field around the dipole with maximum absolute value
ystart = NPmax*Dp - Dp/2
yend = ystart + Dp
zstart = 4*a
zend = -4*a
npointsy = 100
npointsz = 100
dy = (yend - ystart)/npointsy
dz = (zend - zstart)/npointsz

fCD = np.zeros((npointsy, npointsz), dtype=float)
Efield = np.zeros((npointsy, npointsz), dtype=float)
Hfield = np.zeros((npointsy, npointsz), dtype=float)

# Map construction point by point
xj = 0
zj = 0
for indy in range(0, npointsy):
    print(indy)
    for indz in range(0, npointsz):
        x = 0
        y = ystart + indy*dy
        z = zstart + indz*dz

        Escat = np.zeros(3, dtype=complex)
        Hscat = np.zeros(3, dtype=complex)
        Einc = np.zeros(3, dtype=complex)
        Hinc = np.zeros(3, dtype=complex)

        yPmax = NPmax*Dp
        rad = np.sqrt(x**2 + (y - yPmax)**2 + z**2)
        if rad >= a:

            for j in range(1, N+1):
                yj = Dp*(j-1)

                Pj = np.array([P[3*(j-1) + 0], P[3*(j-1) + 1], P[3*(j-1) + 2]])
                Mj = np.array([P[3*(j-1) + 0 + 3*N], P[3*(j-1) + 1 + 3*N],
                               P[3*(j-1) + 2 + 3*N]])

                Escat = Escat + k**2/eps_0*Ge(Pj, k, x, y, z, xj, yj, zj)
                + 1j*Z_0*k**2*Gm(Mj, k, x, y, z, xj, yj, zj)

                Hscat = Hscat - 1j*k**2/(Z_0*eps_0)*Gm(Pj, k, x, y, z, xj, yj,
                                                       zj)
                + k**2*Ge(Mj, k, x, y, z, xj, yj, zj)

            Einc = np.array([EH0(theta, [ncf1, ncf2], k, 'E', 1,
                                 np.array([x, y, z])),
                             EH0(theta, [ncf1, ncf2], k, 'E', 2,
                                 np.array([x, y, z])),
                             EH0(theta, [ncf1, ncf2], k, 'E', 3,
                                 np.array([x, y, z]))])

            Hinc = np.array([EH0(theta, [ncf1, ncf2], k, 'H', 1,
                                 np.array([x, y, z])),
                             EH0(theta, [ncf1, ncf2], k, 'H', 2,
                                 np.array([x, y, z])),
                             EH0(theta, [ncf1, ncf2], k, 'H', 3,
                                 np.array([x, y, z]))])

        Etot = Einc + Escat
        Htot = Hinc + Hscat

        dEH = np.conj(Escat[0])*Hscat[0] + np.conj(Escat[1])*Hscat[1] \
            + np.conj(Escat[2])*Hscat[2]

        dEE = np.conj(Escat[0])*Escat[0] + np.conj(Escat[1])*Escat[1] \
            + np.conj(Escat[2])*Escat[2]

        dHH = np.conj(Hscat[0])*Hscat[0] + np.conj(Hscat[1])*Hscat[1] \
            + np.conj(Hscat[2])*Hscat[2]

        fCD[indy, indz] = -Z_0*np.imag(dEH)
