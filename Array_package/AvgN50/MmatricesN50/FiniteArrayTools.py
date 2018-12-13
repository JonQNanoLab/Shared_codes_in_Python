import numpy as np


def EH0(theta, PolParam, k, EH, comp, rpart):
    """
    This function calculates the components of the incident plane wave of
    wavevector K in a location 'rpart' and returns them as [Ex,Ey,Ez] or
    [Hx,Hy,Hz], depending on the input EH parameter. The function returns the
    value of incident E or H in complex notation!

    Field polarization vectors U1 and U2 fulfill: cros(U1,U2) = k direction.
    So, [U1, U2, K] form a different system of coordinates to that generated
    by XYZ axis. Origin, in both systems, is considered in the center of sphere
    1 (sphere on the left edge of the array), and Y axis (positive) towards
    the center of sphere 2.Z axis is chosen such that it lies in the plane
    formed by, K, the incident wavevector and Y axis, and perpendicular to Y
    axis. X axis always stands 'outside' the screen, which determines
    inequivocally the direction of Z axis using the relation: np.cross(ux,uy)
    = uz. 'theta' is the angle formed by K vector and Z axis, positive if K
    component on Y axis is positive.

    This means that U2 is 'theta' rotated with respect to Y axis and K is
    'theta' rotated which respect to Z axis. U1 stays the same under rotation,
    as it points in the X axis direction, which does not feel the effect of
    the rotation matrix.


        k1                              x
    U = k2 = k1.U1 + k2.U2 + kk.Uk; X = y = x.Ux + y.Uy + z.Uz
        kk                              z

    (Note that: cross(U1,U2) = k direction)

    Definition of rotation matrix:

        1  0  0
    R = 0  c -s  ; c = cos(theta); s = sin(theta)
        0  s  c

    U = R.X  -> X = Inv(R).U = Traspose(R).U = TR.U

    ----- Parameters -----
    theta: incidence angle, float

    PolParam: incidence angle and polarization norm. vectors, [complex,complex]

    k: wavevector modulus, real

    EH: E or H mode, string('E' or 'H')

    comp: component of E or H we want to calculate, int

    rpart: location of the particle, [real, real, real]
    """

    assert EH == 'E' or EH == 'H', 'You didnt introduce a valid field name!'
    assert type(k) != complex, 'You cannot introduce complex values for k!'

    # Vacuum constants
    eps_0 = 8.854187817*10**(-12)
    mu_0 = 4*np.pi*10**(-7)
    Z_0 = np.sqrt(mu_0/eps_0)

    # Amplitude of incident EM field
    E0 = 1

    # Polarization coefficients
    ncf1 = PolParam[0]      # Normalized coefficient of component U1
    ncf2 = PolParam[1]      # Normalized coefficient of component U2

    K = k*np.array([0, np.sin(theta), np.cos(theta)])   # Incident wavevector

    # Calculate the normalized incident field polarization vectors
    U_E = np.array([ncf1, ncf2, 0])
    U_H = np.array([-ncf2, ncf1, 0])
    TR = np.array([[1, 0, 0], [0, np.cos(theta), np.sin(theta)],
                   [0, -np.sin(theta), np.cos(theta)]])

    # Transform elec. polarization vector to (X,Y,Z) coordinate system and
    # amplitude calculation.
    pol_E = TR.dot(U_E)
    E_XYZ = E0*pol_E

    # Transform mag. polarization vector to (X,Y,Z) coordinate system and
    # amplitude calculation.
    pol_H = TR.dot(U_H)
    H_XYZ = E0/Z_0*pol_H

    # Phase factor. Dot product in Python does not conjugate K vector.
    phi = np.dot(K, rpart)

    if EH == 'E':
        if comp == 1:
            return E_XYZ[0]*np.exp(1j*phi)

        if comp == 2:
            return E_XYZ[1]*np.exp(1j*phi)

        if comp == 3:
            return E_XYZ[2]*np.exp(1j*phi)

    if EH == 'H':
        if comp == 1:
            return H_XYZ[0]*np.exp(1j*phi)

        if comp == 2:
            return H_XYZ[1]*np.exp(1j*phi)

        if comp == 3:
            return H_XYZ[2]*np.exp(1j*phi)


def PSI(x, n, fd):
    """
    This function calculates the PSI Ricatti-Bessel function given in Bohren
    page 101.

    ----- Parameters -----

    x: variable of Ricatti-Bessel function, complex

    n: order of the Ricatti-Bessel function, int

    fd: function or derivative mode, string('f' or 'd')
    """

    assert type(n) == int and n > 0, 'The order, n, is not valid!'
    assert type(fd) == str, 'fd variable needs either f or d string value!'

    # Calculus of the spherical Bessel function of order n
    bessel_j0 = np.sin(x)/x
    bessel_j1 = np.sin(x)/x**2 - np.cos(x)/x

    if n == 0:
        bessel_jn = bessel_j0

    if n == 1:
        bessel_jn = bessel_j1

    if n > 1:
        cont = 2
        z_n1 = bessel_j1
        z_n2 = bessel_j0

        while cont <= n:
            bessel_jn = (2*cont - 1)*z_n1 - z_n2
            cont += 1
            z_n2 = z_n1
            z_n1 = bessel_jn

    # Calculus of the derivative of the spherical Bessel function of order n
    if fd == 'd':

        d_bessel_j0 = -bessel_j1

        if n == 0:
            d_bessel_jn = d_bessel_j0

        if n == 1:
            d_bessel_jn = bessel_j0 - (n+1)/x*bessel_j1

        else:
            d_bessel_jn = z_n2 - (n+1)/x*z_n1

    # Return function or derivative
    if fd == 'f':
        return x*bessel_jn

    else:
        return bessel_jn + x*d_bessel_jn


def KSI(x, n, fd):
    """
    This function calculates the KSI Ricatti-Bessel function given in Bohren
    page 101.

    ----- Parameters -----
    x: variable of Ricatti-Bessel function, complex

    n: order of the Ricatti-Bessel function, int

    fd: function or derivative mode, string('f' or 'd')
    """

    assert type(n) == int and n > 0, 'The order, n, is not valid!'
    assert type(fd) == str, 'fd variable needs either f or d string value!'

    # Calculus of the spherical Hankel function of order n
    bessel_h0 = np.sin(x)/x - 1j*(np.cos(x)/x)
    bessel_h1 = np.sin(x)/x**2 - np.cos(x)/x - 1j*(np.cos(x)/x**2
                                                   + np.sin(x)/x)

    if n == 0:
        bessel_hn = bessel_h0

    if n == 1:
        bessel_hn = bessel_h1

    if n > 1:
        cont = 2
        z_n1 = bessel_h1
        z_n2 = bessel_h0

        while cont <= n:
            bessel_hn = (2*cont - 1)*z_n1 - z_n2
            cont += 1
            z_n2 = z_n1
            z_n1 = bessel_hn

    # Calculus of the derivative of the spherical Bessel function of order n
    if fd == 'd':

        d_bessel_h0 = -bessel_h1

        if n == 0:
            d_bessel_hn = d_bessel_h0

        if n == 1:
            d_bessel_hn = bessel_h0 - (n+1)/x*bessel_h1

        else:
            d_bessel_hn = z_n2 - (n+1)/x*z_n1

    # Return function or derivative
    if fd == 'f':
        return x*bessel_hn

    else:
        return bessel_hn + x*d_bessel_hn


def ab_MieCoef(a, LambdaArray, EpsParticle, EpsEmbedded, N):
    """
    This function calculates a and b coefficients from Mie theory, following
    expressions (4.56) and (4.57) from Bohren of a given order N. It is thought
    to calculate a1 and b1 parameters, although, in principle, it could be used
    for any order.

    ----- Parameters -----
    a: radius of the sphere, float

    LambdaArray: array of wavelengths for which electric permittivity of the
    particle (given in EpsParticle) is defined, real list

    EpsParticle: electric permittivity of the particle for a range of
    wavelengths, complex list

    EpsEmbedded: electric permittivity of the medium in which the particle
    is embedded, complex list

    N: order of Mie theory for which the calculation is going to be made, int
    """

    assert len(LambdaArray) == len(EpsParticle), 'Dimension of input \
                                                 EpsParticle is not adequate \
                                                 in ab_MieCoef!'
    assert len(LambdaArray) == len(EpsEmbedded), 'Dimension of input \
                                                 EpsEmbedded is not adequate \
                                                 in ab_MieCoef!'

    A_Mie = np.zeros(len(LambdaArray), dtype=complex)
    B_Mie = np.zeros(len(LambdaArray), dtype=complex)

    for k in range(0, len(LambdaArray)):
        x = 2*np.pi*np.sqrt(EpsEmbedded[k])*a/LambdaArray[k]
        m = np.sqrt(EpsParticle[k]/EpsEmbedded[k])

        A_Mie[k] = (m*PSI(m*x, N, 'f')*PSI(x, N, 'd') - PSI(x, N, 'f')
                    * PSI(m*x, N, 'd')) / (m*PSI(m*x, N, 'f')*KSI(x, N, 'd')
                                           - KSI(x, N, 'f')*PSI(m*x, N, 'd'))

        B_Mie[k] = (PSI(m*x, N, 'f')*PSI(x, N, 'd') - m*PSI(x, N, 'f')
                    * PSI(m*x, N, 'd')) / (PSI(m*x, N, 'f')*KSI(x, N, 'd')
                                           - m*KSI(x, N, 'f')*PSI(m*x, N, 'd'))

    return A_Mie, B_Mie


def lambda_kerker(a, LambdaArray, EpsParticle, EpsEmbedded, N):
    """
    This function calculates a wavelength for which the Kerker's first
    condition is fulfilled in a given spherical particle characterized by
    its electric permittivity. Kerker's first condition is defined:
    a_e(lambda) = a_m(lambda).

    First we obtain a1 and b1 coefficients from function ab_MieCoef and then
    we impose the aforementioned condition.

    ----- Parameters -----
    a: radius of the sphere considered, float

    LambdaArray: array of wavelengths for which electric permittivity of the
    particle (given in EpsParticle) is defined, real list

    EpsParticle: electric permittivity of the particle for a range of
    wavelengths, complex list

    EpsEmbedded: electric permittivity of the medium in which the particle
    is embedded, complex list

    N: order of Mie theory for which the calculation is going to be made, int
    """

    assert len(LambdaArray) == len(EpsParticle), 'Dimension of input \
                                                  EpsParticle is not adequate \
                                                  in lambda_Kerker!'
    assert len(LambdaArray) == len(EpsEmbedded), 'Dimension of input \
                                                  EpsEmbedded is not adequate \
                                                  in lambda_Kerker!'

    k = 2*np.pi/LambdaArray
    a1, b1 = ab_MieCoef(a, LambdaArray, EpsParticle, EpsEmbedded, N=1)

    a_e = 1j*(6*np.pi/k**3)*a1
    a_m = 1j*(6*np.pi/k**3)*b1

    # Functions of real and imaginary parts of the polarizabilities,
    # that must be 0.
    fRe = np.real(a_e) - np.real(a_m)
    fIm = np.imag(a_e) - np.imag(a_m)

    # Initialize lists to save wavelength's for which a_e = a_m is fulfilled
    LambdaK = []
    a_eK = []
    a_mK = []

    cont = 0
    for i in range(0, len(LambdaArray)-1):
        signcheck1 = fRe[i]*fRe[i+1]
        signcheck2 = fIm[i]*fIm[i+1]

        if signcheck1 < 0 and signcheck2 < 0:
            cont += 1
            LambdaK.append(LambdaArray[i])
            a_eK.append(a_e[i])
            a_mK.append(a_m[i])

    return np.array(LambdaK), np.array(a_eK), np.array(a_mK)
