import numpy as np
import scipy.io as sio
from FiniteArrayTools import EH0, ab_MieCoef, lambda_kerker
from GreenFunc_r import Ge, Gm

Psupra = sio.loadmat('Psupra.mat')['Psupra']
kd = 71  # resonance for N=50
ktheta = -1  # last element
P = P = Psupra[kd][ktheta][:]
Dp = 300e-9 + 4.5e-9*kd
theta_i = np.pi/2
R = 151e-9

# ---------- CONSTANTS -------------------------------------------------- #
# ----------------------------------------------------------------------- #
nm = 10**(-9)
eps_0 = 8.854187817*10**(-12)
mu_0 = 4*np.pi*10**(-7)
Z_0 = np.sqrt(mu_0/eps_0)

# ---------- PARTICLE PERMITTIVITY -------------------------------------- #
# ----------------------------------------------------------------------- #

# Read the data of the elec. permittivity of the particles.
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

# ---------- MAIN PARAMETERS OF THE PROBLEM ----------------------------- #
# ----------------------------------------------------------------------- #

# Number of nanoparticles
N = 50

# Radius of the nanoparticles
a = 150*nm

# Definition of the incident EM field
Kerker = True
cf1 = 1         # Polarization coefficient in the U1 direction (not norm.)
cf2 = +1j       # Polarization coefficient in the U1 direction (not norm.)

norm = np.sqrt(cf1*np.conj(cf1) + cf2*np.conj(cf2))
ncf1 = cf1/norm  # Polarization coefficient in the U1 direction (norm.)
ncf2 = cf2/norm  # Polarization coefficient in the U2 direction (norm.)

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

# ---------- OPTIONS OF THE FUNCTION ------------------------------------ #
# ----------------------------------------------------------------------- #

# Define resolution for sph. integration in both variables (theta, phi)
res_ss = 30

# Define (theta, phi) values linspace
theta_ss = np.linspace(0, np.pi, res_ss)
dtheta = np.abs(theta_ss[0] - theta_ss[1])
phi_ss = np.linspace(0, 2*np.pi, res_ss)
dphi = np.abs(phi_ss[0] - phi_ss[1])

# 2D matrices Theta and Phi, contain theta and phi values of every point.
Theta_ss, Phi_ss = np.meshgrid(theta_ss, phi_ss, indexing='ij')

# Make this matrices have a shape of an array
ttheta_ss = np.reshape(Theta_ss, (res_ss * res_ss, 1))
pphi_ss = np.reshape(Phi_ss, (res_ss * res_ss, 1))

fCDsum = 0
# First we need to place ourselves in all the spheres
nrange = range(0, N)

for n in nrange:
    print(n)

    # Convert from spherical to cartesian coordinates
    xx = R*np.sin(ttheta_ss)*np.cos(pphi_ss)
    yy = R*np.sin(ttheta_ss)*np.sin(pphi_ss) + n*Dp  # CHECK n*Dp term!!
    zz = R*np.cos(ttheta_ss)
    rpart = np.array([xx[:, 0], yy[:, 0], zz[:, 0]])

    # Calculate the incident E field in (R, theta, phi) point
    Einc = np.array([EH0(theta_i, [ncf1, ncf2], k, 'E', 1, rpart),
                     EH0(theta_i, [ncf1, ncf2], k, 'E', 2, rpart),
                     EH0(theta_i, [ncf1, ncf2], k, 'E', 3, rpart)])

    # Calculate the incident H field in (R, theta, phi) point
    Hinc = np.array([EH0(theta_i, [ncf1, ncf2], k, 'H', 1, rpart),
                     EH0(theta_i, [ncf1, ncf2], k, 'H', 2, rpart),
                     EH0(theta_i, [ncf1, ncf2], k, 'H', 3, rpart)])

    # Loop to calculate scattered E, H fields in (R, theta, phi point
    Escat = np.zeros((3, 900), dtype=complex)
    Hscat = np.zeros((3, 900), dtype=complex)
    xj = 0
    zj = 0
    for j in range(1, N+1):
        yj = Dp*(j-1)
        r = np.sqrt((xx[:, 0] - xj)**2 + (yy[:, 0] - yj)**2
                    + (zz[:, 0] - zj)**2)

        UR = np.array([xx[:, 0], yy[:, 0] - yj, zz[:, 0]]) / r

        Pj = np.array([P[3*(j-1) + 0], P[3*(j-1) + 1],
                       P[3*(j-1) + 2]])
        Mj = np.array([P[3*(j-1) + 0 + 3*N], P[3*(j-1) + 1 + 3*N],
                       P[3*(j-1) + 2 + 3*N]])

        Escat = Escat \
            + k**2/eps_0*Ge(Pj, k, UR, r) \
            + 1j*Z_0*k**2*Gm(Mj, k, UR, r)

        Hscat = Hscat \
            - 1j*k**2/(Z_0*eps_0)*Gm(Pj, k, UR, r) \
            + k**2*Ge(Mj, k, UR, r)

    Etot = Einc + Escat
    Htot = Hinc + Hscat

    dEH = np.conj(Etot[0, :])*Htot[0, :] + np.conj(Etot[1, :])*Htot[1, :] \
        + np.conj(Etot[2, :])*Htot[2, :]

    # fCD value over each point in a sphere of radius R
    fCD = -Z_0*np.imag(dEH)

    # function we are actually integrating
    fCD_integ = 1/(4*np.pi)*fCD*np.sin(ttheta_ss[:, 0])

    # Reshape the integrating function to better manage the computation
    FCD_integ = np.reshape(fCD_integ, (res_ss, res_ss))

    # Integral in theta variable
    fCDint_theta = np.trapz(FCD_integ, theta_ss, dtheta)

    # Integral in phi variable
    fCDint = np.trapz(fCDint_theta, phi_ss, dphi)

    # sum fCD per sphere
    fCDsum += fCDint

print(fCDsum/len(nrange))
