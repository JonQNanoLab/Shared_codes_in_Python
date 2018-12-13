#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  3 08:38:33 2018

@author: jon
"""

import numpy as np
import multiprocessing as mp
import scipy.io as sio  # To save in .mat format
from tqdm import tqdm  # To control long loops' flow
from MainTools import AGmatrix_array
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


# ---------- DIPOLE POSITION DATA ORIGIN ------------------------------------ #
# --------------------------------------------------------------------------- #

# The program allows for reading the spatial distribution of dipoles either
# from a file 'DipolePos.txt' or directly using functions for a given geometry.
# Variable Fromfile determines which of both ways we are using.

Fromfile = False


# ---------- MAIN PARAMETERS OF THE PROBLEM --------------------------------- #
# --------------------------------------------------------------------------- #

# Number of nanoparticles
N = 200

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


# ---------------- CALCULATE M MATRIX --------------------------------------- #
# --------------------------------------------------------------------------- #

Dplist = np.linspace(2*a, 3000*nm, 601)


def func(Dp):

    # For each value of the Dp value, we calculate the dipole coupling matrix
    M = AGmatrix_array(Dp, N, a_e, a_m, k)

    return M, Dp


# Generate processes equal to the number of cores
ncore = 8
print('\n\n POOL HAS BEEN OPENED!')
pool = mp.Pool(processes=ncore)

with tqdm(total=len(Dplist)) as pbar:
    # Distribute the parameter sets evenly across the cores
    for i, res in enumerate(pool.imap(func, Dplist), 0):

        sio.savemat('Mmatrix' + str(N) + 'spheres' + str(round(res[1]/nm, 1))
                    + 'nm.mat', {'M': res[0]})

        pbar.update(1)

    pbar.close()
    pool.close()

print('\n\n POOL HAS BEEN CLOSED!')
