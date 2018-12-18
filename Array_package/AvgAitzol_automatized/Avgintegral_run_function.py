import numpy as np
from FiniteArrayTools import EH0
from GreenFunc_r import Ge, Gm

# ---------- CONSTANTS ------------------------------------------------------ #
# --------------------------------------------------------------------------- #

nm = 10**(-9)
eps_0 = 8.854187817*10**(-12)
mu_0 = 4*np.pi*10**(-7)
Z_0 = np.sqrt(mu_0/eps_0)
clight = 1/np.sqrt(eps_0*mu_0)


def Avgintegral_run_function(P, Dp, R, EHinc_param):
    """
    This function calculates the average integral of fCD around a given number
    of spheres.

    ----- Parameters -----
    P: dipole distribution, 6N dimensional complex array

    Dp: distance among dipoles, float

    R: radius at which integration is done, float

    EHinc_param: parameters of the incident EM field, complex list
    """
    theta_i = EHinc_param[0]
    [ncf1, ncf2] = EHinc_param[1]
    k = EHinc_param[2]
    N = int(len(P)/6)

    # Define resolution for sph. integration in both variables (theta, phi)
    res_ss = 30

    # Define (theta, phi) values linspace
    theta_ss = np.linspace(0, np.pi, res_ss)
    dtheta = np.abs(theta_ss[0] - theta_ss[1])

    phi_ss = np.linspace(0, 2*np.pi, res_ss)
    dphi = np.abs(phi_ss[0] - phi_ss[1])

    # 2D matrices Theta_ss and Phi_ss, contain all theta and phi paired values.
    #
    # Important note:
    # Theta_ss saves theta values in rows
    # Phi_ss saves phi values in columns
    Theta_ss, Phi_ss = np.meshgrid(theta_ss, phi_ss, indexing='ij')

    # Make this matrices have a shape of an array
    ttheta_ss = np.reshape(Theta_ss, (res_ss * res_ss, 1))
    pphi_ss = np.reshape(Phi_ss, (res_ss * res_ss, 1))

    # Initialize the value of the fCD average value
    fCDsum = 0

    # First we need to place ourselves in all the spheres
    nrange = range(0, N)
    for n in nrange:

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

        # Function we are actually integrating
        fCD_integ = 1/(4*np.pi)*fCD*np.sin(ttheta_ss[:, 0])

        # Reshape the integrating function to better manage the computation
        FCD_integ = np.reshape(fCD_integ, (res_ss, res_ss))

        # Integral in theta variable. Integral across rows, as Theta_ss
        # saved the theta values in that dimension.
        fCDint_theta = np.trapz(FCD_integ, theta_ss, dtheta, axis=0)

        # Integral in phi variable
        fCDint = np.trapz(fCDint_theta, phi_ss, dphi)

        # Sum fCD per sphere
        fCDsum += fCDint

    return fCDsum
