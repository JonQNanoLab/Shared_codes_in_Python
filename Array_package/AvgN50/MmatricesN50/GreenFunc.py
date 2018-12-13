import numpy as np


def Ge(V, k, x, y, z, xj, yj, zj):
    """
    This function calculates the dyadic product between the electric dyadic
    Green's function and a given dipole vector.

    Input:
        V: dipole column vector
        k: wavelength
        x,y,z: coordinates where the field is being evaluated
        xj,yj,zj: coordinates where the field is being generated

    Output:
        Dyadic product Ge.V
    """

    assert V.shape == (3, 1) or V.shape == (3,), 'V in Ge not correct dim!'

    R = np.array([x-xj, y-yj, z-zj])
    r = np.sqrt((x-xj)**2 + (y-yj)**2 + (z-zj)**2)
    UR = R/r
    kr = k*r

    # Diagonal and non-diagonal terms of the electric dyadic Green's function
    d = 1 + 1j/kr - 1/kr**2
    nd = -1 - 3j/kr + 3/kr**2

    # Projection between UR and V vectors
    proy = np.dot(np.transpose(UR), V)

    # Scalar Green's function
    g = np.exp(1j*kr)/(4*np.pi*r)

    return g*(d*V + nd*proy*UR)


def Gm(V, k, x, y, z, xj, yj, zj):

    """
    This function calculates the dyadic product between the magnetic dyadic
    Green's function and a given dipole vector.

    Input:
        V: dipole column vector
        k: wavelength
        x,y,z: coordinates where the field is being evaluated
        xj,yj,zj: coordinates where the field is being generated

    Output:
        Dyadic product Gm.V
    """

    assert V.shape == (3, 1) or V.shape == (3,), 'V in Gm not correct dim!'

    R = np.array([x-xj, y-yj, z-zj])
    r = np.sqrt((x-xj)**2 + (y-yj)**2 + (z-zj)**2)
    UR = R/r
    kr = k*r

    # Only term of the magnetic dyadic Green's function
    f = 1j - 1/kr

    # Cross product between UR and V
    CR = np.cross(UR, V)

    # Scalar Green's function
    g = np.exp(1j*kr)/(4*np.pi*r)

    return f*g*CR
