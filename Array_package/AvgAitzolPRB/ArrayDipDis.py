import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from MainTools import EHinc_array, AGmatrix_array
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
N = 40

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
Dp = 395*nm

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

    mx = P[3*(i + N)]
    my = P[3*(i + N) + 1]
    mz = P[3*(i + N) + 2]

    Pabs[i] = np.sqrt(px*np.conj(px) + py*np.conj(py) + pz*np.conj(pz))
    Mabs[i] = np.sqrt(mx*np.conj(mx) + my*np.conj(my) + mz*np.conj(mz))

    Px[i] = np.abs(px)*np.cos(np.angle(px))
    Py[i] = np.abs(py)*np.cos(np.angle(py))
    Pz[i] = np.abs(pz)*np.cos(np.angle(pz))

    Mx[i] = np.abs(mx)*np.cos(np.angle(mx))
    My[i] = np.abs(my)*np.cos(np.angle(my))
    Mz[i] = np.abs(mz)*np.cos(np.angle(mz))


P0 = eps_0*a_e
M0 = a_m/Z_0

fig1 = plt.figure(1)
ax1 = fig1.add_subplot(111, projection='3d')
ax1.quiver(np.zeros(N)/(10**3*nm), np.arange(0, N)*Dp/nm, np.zeros(N)/nm,
           Px/P0, Py/P0, Pz/P0, length=0.02)
plt.xlabel('X axis (nm)')
plt.ylabel('Y axis (um)')
plt.title('Elec. dipole distribution, N = ' + str(N))

fig = plt.figure(2)
ax = fig.add_subplot(111, projection='3d')
ax.quiver(np.zeros(N)/(10**3*nm), np.arange(0, N)*Dp/nm, np.zeros(N)/nm,
          Mx/M0, My/M0, Mz/M0, length=0.02)
plt.xlabel('X axis (nm)')
plt.ylabel('Y axis (um)')
plt.title('Magn. dipole distribution, N = ' + str(N))
plt.show()
