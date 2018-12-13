import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from FiniteArrayTools import EH0, ab_MieCoef
from MainTools import EHinc_array, AGmatrix_array
from GreenFunc import Ge, Gm


# def AvgIntegral(P, Dp, theta, R):
"""
This function calculates the average fCD integral over a number N of
spheres defined by the total (electric and magnetic) dipolar moments P,
under an illumination conditions determined by Dp and theta.

----- Parameters -----
P: dipolar moments induced and already calculated, complex Dplist

Dp: distance between spheres, float

theta: angle of incidence, float
"""
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
Kerker = False
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

# Discretization values for the integral of fCD
# Theta
ntheta = 40
dtheta = np.pi/ntheta
# Phi
nphi = 60
dphi = 2*np.pi/nphi

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

    # Calculus independent of Lambda, proper calculus
    EHvec = EHinc_array(N, Dp, theta_i, [ncf1, ncf2], k, 'PW')
    M = AGmatrix_array(Dp, N, a_e, a_m, k)
    P = M.dot(EHvec)

    fCDsum = 0
    # First we need to place ourselves in all the spheres
    for n in range(0, N):
        # For each sphere we have to do the integration over theta, phi values
        for ktheta in range(0, ntheta + 1):
            theta = dtheta*ktheta

            for kphi in range(0, nphi + 1):
                phi = dphi*kphi

                # For each point in space (R, theta, phi) incident and scatte-
                # red fields must be calculated

                # Convert from spherical to cartesian coordinates
                x = R*np.sin(theta)*np.cos(phi)
                y = R*np.sin(theta)*np.sin(phi) + n*Dp  # CHECK n*Dp term!!
                z = R*np.cos(theta)
                rpart = [x, y, z]

                # Calculate the incident E field in such (R, theta, phi) point
                Einc = np.array([EH0(theta_i, [ncf1, ncf2], k, 'E', 1, rpart),
                                 EH0(theta_i, [ncf1, ncf2], k, 'E', 2, rpart),
                                 EH0(theta_i, [ncf1, ncf2], k, 'E', 3, rpart)])

                # Calculate the incident H field in such (R, theta, phi) point
                Hinc = np.array([EH0(theta_i, [ncf1, ncf2], k, 'H', 1, rpart),
                                 EH0(theta_i, [ncf1, ncf2], k, 'H', 2, rpart),
                                 EH0(theta_i, [ncf1, ncf2], k, 'H', 3, rpart)])

                # Loop to calculate scattered E, H fields in (R, theta, phi)
                Escat = np.zeros(3, dtype=complex)
                Hscat = np.zeros(3, dtype=complex)
                xj = 0
                zj = 0
                for j in range(1, N+1):
                    yj = Dp*(j-1)

                    Pj = np.array([P[3*(j-1) + 0], P[3*(j-1) + 1],
                                   P[3*(j-1) + 2]])
                    Mj = np.array([P[3*(j-1) + 0 + 3*N], P[3*(j-1) + 1 + 3*N],
                                   P[3*(j-1) + 2 + 3*N]])

                    Escat[:] = Escat[:] \
                        + k**2/eps_0*Ge(Pj, k, x, y, z, xj, yj, zj) \
                        + 1j*Z_0*k**2*Gm(Mj, k, x, y, z, xj, yj, zj)

                    Hscat[:] = Hscat[:] \
                        - 1j*k**2/(Z_0*eps_0)*Gm(Pj, k, x, y, z, xj, yj, zj) \
                        + k**2*Ge(Mj, k, x, y, z, xj, yj, zj)

                Etot = Einc + Escat
                Htot = Hinc + Hscat

                dEH = np.conj(Etot[0])*Htot[0] + np.conj(Etot[1])*Htot[1] \
                    + np.conj(Etot[2])*Htot[2]

                fCD = -Z_0*np.imag(dEH)
                fCDsum = fCDsum + 1/(4*np.pi)*fCD*np.sin(theta)*dtheta*dphi

    fCDsum_tot[kwave] = fCDsum


sio.savemat('fCDavg.mat', {'fCDavg': fCDsum_tot/N})

plt.plot(LambdaArray/nm, fCDsum_tot, '.')
plt.xlabel('Wavelength (nm)')
plt.ylabel('fCD_avg')
plt.title('Average CD enhancement')
plt.xlim((400, 800))
plt.show()
# return fCDsum
