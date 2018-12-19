import numpy as np


def Ge(V, k, UR, r):
    """
    ---------------------------------------------------------------------------

    CHECK!! FUNCTION IS ADAPTED FOR meshgrid AND reshape FUNCTIONS.

    ---------------------------------------------------------------------------
    This function calculates the dyadic product between the electric dyadic
    Green's function and a given dipole vector.

    ----- Parameters -----
    V: dipole column vector, [complex, complex, complex]

    k: wavevector modulus, real

    x,y,z: coordinates where the field is being evaluated

    xj,yj,zj: coordinates where the field is being generated

    """
    # assert V.shape == (3, 1) or V.shape == (3,), 'V in Ge not correct dim!'
    kr = k*r

    # Diagonal and non-diagonal terms of the electric dyadic Green's function
    d = 1 + 1j/kr - 1/kr**2
    nd = -1 - 3j/kr + 3/kr**2

    # Projection between UR and V vectors
    proy = np.dot(V, UR)

    # Scalar Green's function
    g = np.exp(1j*kr)/(4*np.pi*r)

    # Gx
    Gx = g * (d * V[0] + nd * proy * UR[0, :])
    Gy = g * (d * V[1] + nd * proy * UR[1, :])
    Gz = g * (d * V[2] + nd * proy * UR[2, :])

    return np.array([Gx, Gy, Gz])


def Gm(V, k, UR, r):
    """
    ---------------------------------------------------------------------------

    CHECK!! FUNCTION IS ADAPTED FOR meshgrid AND reshape FUNCTIONS.

    ---------------------------------------------------------------------------
    This function calculates the dyadic product between the magnetic dyadic
    Green's function and a given dipole vector.

    ----- Parameters -----
    V: dipole column vector, [complex, complex, complex]

    k: wavevector modulus, real

    x,y,z: coordinates where the field is being evaluated

    xj,yj,zj: coordinates where the field is being generated
    """
    assert V.shape == (3, 1) or V.shape == (3,), 'V in Gm not correct dim!'

    kr = k*r

    # Only term of the magnetic dyadic Green's function
    f = 1j - 1/kr

    # Cross product between UR and V
    CR = np.transpose(np.cross(UR, V, axisa=0))

    # Scalar Green's function
    g = np.exp(1j*kr)/(4*np.pi*r)

    return f*g*CR
