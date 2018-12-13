import numpy as np
from FiniteArrayTools import EH0
from GreenFunc import Ge, Gm


# ---------- TOOLS FOR: ARRAY GEOMETRY -------------------------------------- #
# --------------------------------------------------------------------------- #

def EHinc_array(N, Dp, theta, PolParam, k, mode):
    """
    This function calculates the value of a given illuminating field at every
    dipole site of a chosen geometry.

    ----- Parameters -----
    N: number of spheres, integer

    Dp: distance from the center of one sphere to another, float

    theta: incidence angle, float

    PolParam: polarization norm. vectors, [complex, complex]

    k: wavevector modulus, real

    mode: type of incident field (currently only 'PW' available)
    """
    # Polarization coefficients
    ncf1 = PolParam[0]      # Normalized coefficient of component U1
    ncf2 = PolParam[1]      # Normalized coefficient of component U2

    # Initialize the vector that we will finally return
    EHvec = np.zeros(6*N, dtype=complex)

    # Plane-wave mode
    if mode == 'PW':

        # We run through all the first 3N elements of EHvec (electric fields)
        for ind in range(0, 3*N):
            j = np.floor(ind/3) + 1             # Particle index (1 -> N)
            j = int(j)                          # Convert j to an integer
            ind_c = int((ind+1) - (j-1)*3)      # Component index

            # Position vector of particle j
            r_j = np.array([0, Dp*(j-1), 0])

            # Calculus of the electric components of EHvec
            EHvec[ind] = EH0(theta, [ncf1, ncf2], k, 'E', ind_c, r_j)

            # Calculus of the magnetic components of EHvec
            EHvec[ind + 3*N] = EH0(theta, [ncf1, ncf2], k, 'H', ind_c, r_j)

    return EHvec


def C_Ext_array(N, Dp, P, theta, PolParam, k):
    """
    This function calculates the extinction cross section once the dipole
    geometry/distribution has been set up.

    ----- Parameters -----
    N: number of spheres, integer

    Dp: distance from the center of one sphere to another, float

    P: already calculated dipole distribution, 6N dimensional complex array

    theta: incidence angle, float

    PolParam: polarization norm. vectors, [complex, complex]

    k: wavevector modulus, real
    """
    # Polarization coefficients
    ncf1 = PolParam[0]      # Normalized coefficient of component U1
    ncf2 = PolParam[1]      # Normalized coefficient of component U2

    # Polarization vectors
    U1 = np.transpose(np.array([1, 0, 0]))
    U2 = np.transpose(np.array([0, np.cos(theta), -np.sin(theta)]))

    # Definition of the incident polarization vector, NE (Novotny)
    NE = ncf1*U1 + ncf2*U2

    # Vacuum constants
    eps_0 = 8.854187817*10**(-12)
    mu_0 = 4*np.pi*10**(-7)
    Z_0 = np.sqrt(mu_0/eps_0)

    # Define the point in which we are going to calculate extinction
    Rfar = np.array([0, np.sin(theta), np.cos(theta)])*(10**6 * N * Dp)
    rx = Rfar[0]
    ry = Rfar[1]
    rz = Rfar[2]

    # Radial distance as defined in Novotny, Chapter 15
    Rradial = np.sqrt((rx)**2 + (ry)**2 + (rz)**2)

    # Initialize far field electric field vector
    Efar = np.zeros(3)

    # Calculate the field radiated in the far field by all the dipoles
    xj = 0
    zj = 0
    for j in range(1, N+1):
        yj = Dp*(j-1)

        Pj = np.array([P[3*(j-1) + 0], P[3*(j-1) + 1], P[3*(j-1) + 2]])
        Mj = np.array([P[3*(j-1) + 0 + 3*N], P[3*(j-1) + 1 + 3*N],
                       P[3*(j-1) + 2 + 3*N]])

        Efar = Efar + k**2/eps_0*Ge(Pj, k, rx, ry, rz, xj, yj, zj) \
            + 1j*Z_0*k**2*Gm(Mj, k, rx, ry, rz, xj, yj, zj)

    # Far-field X vector (Novotny), considering the amplitude of incident field
    # E0 = 1 and extinction cross-section  calculus
    X = (-1j*k*Rradial)*np.exp(-1j*k*Rradial)*Efar
    CExt = 4*np.pi/k**2*np.real(np.dot(np.conj(X), NE))

    return CExt


def AGmatrix_array(Dp, N, a_e, a_m, k):
    """
    This function calculates the matrix that solves the 6N dimensional system
    of equations that represent the self-consistent coupling of N electric and
    magnetic dipoles equally spaced in an array. The distance between dipoles
    is defined as Dp and N is the number of dipoles.

    ----- Parameters -----
    Dp: distance between dipoles in the array, float

    N: number of particles, int

    a_e: electric polarizability of the particles, complex

    a_m: magnetic polarizability of the particles, complex

    k: wavevector modulus, float
    """
    # Vacuum constants
    eps_0 = 8.854187817*10**(-12)
    mu_0 = 4*np.pi*10**(-7)
    Z_0 = np.sqrt(mu_0/eps_0)

    # Definition of identity matrix
    Id = np.eye(6*N, 6*N)

    # Definition of the polarizability matrices
    Alpha_e = a_e*np.eye(3*N, 3*N)
    Alpha_m = a_m*np.eye(3*N, 3*N)
    Alpha = np.block([
            [eps_0*Alpha_e, np.zeros((3*N, 3*N))],
            [np.zeros((3*N, 3*N)), Alpha_m]
            ])

    # Initialization of G matrix
    G_e = np.zeros((3*N, 3*N), dtype=complex)
    G_m = np.zeros((3*N, 3*N), dtype=complex)

    # Definition of cartesian unitary vectors
    Ux = np.array([1, 0, 0])
    Uy = np.array([0, 1, 0])
    Uz = np.array([0, 0, 1])

    # In the case of an array placed in the OY axis, the X and Y coordinates
    # of the particles are constant and equal to 0.
    xi = 0.0
    zi = 0.0
    xj = 0.0
    zj = 0.0

    # Calculus of G matrix
    for kind in range(0, 3*N):
        for lind in range(0, 3*N):

            i = np.floor(kind/3) + 1  # Particle's horizontal index
            j = np.floor(lind/3) + 1  # Particle's vertical index
            i = int(i)            # Convert i to an integer
            j = int(j)            # Convert j to an integer

            if i != j:
                # We define the Y coordinates of the dipoles in the array
                yi = Dp*(i-1)
                yj = Dp*(j-1)

                # We define G's small matrix index
                k_s = kind - (i-1)*3
                l_s = lind - (j-1)*3

                # G_e matrix
                if k_s == 0:
                    V = Ge(Ux, k, xj, yj, zj, xi, yi, zi)

                if k_s == 1:
                    V = Ge(Uy, k, xj, yj, zj, xi, yi, zi)

                if k_s == 2:
                    V = Ge(Uz, k, xj, yj, zj, xi, yi, zi)

                G_e[lind, kind] = V[l_s]

                # G_m matrix
                if k_s == 0:
                    W = Gm(Ux, k, xj, yj, zj, xi, yi, zi)

                if k_s == 1:
                    W = Gm(Uy, k, xj, yj, zj, xi, yi, zi)

                if k_s == 2:
                    W = Gm(Uz, k, xj, yj, zj, xi, yi, zi)

                G_m[lind, kind] = W[l_s]

    Amat = k**2/eps_0*G_e
    Bmat = 1j*Z_0*k**2*G_m
    Cmat = -1j*k**2/Z_0*1/eps_0*G_m
    Dmat = k**2*G_e

    G = np.block([
            [Amat, Bmat],
            [Cmat, Dmat]
            ])

    # Finally we return the outcome matrix: M = inv(I - Alpha*G)*Alpha
    return np.linalg.inv(Id - Alpha.dot(G)).dot(Alpha)
