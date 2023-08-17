import edrixs
import numpy as np
from numpy.linalg import matrix_power

def cf_stevens(J, B):
    """
    Given the LS Stevens parameters, return the crystal field matrix in the Ylm basis.

    Parameters
    ----------
    J : integer or half integer
        angular momentum.
    B : dictionary
        Stevens parameters

    Returns
    -------
    cf: 2d complex array, shape=(N, N)
        The matrix form of crystal field Hamiltonian in (J, Jz) basis for fixed J.
        Where N = 2*J+1.

    Notes
    -----
    See appendix B of Ref. [1]_ for further details.

    References
    ----------
    .. [1] McPhase USERS MANUAL https://www2.cpfs.mpg.de/~rotter/homepage_mcphase/manual/node132.html
    """
    
    njs = int(2 * J + 1)

    # define J_z and raising/lowering operators
    Jz = edrixs.get_lz(J)
    Jadd = edrixs.get_ladd(J)
    Jminus = edrixs.get_lminus(J)

    # define other operators up to the power 6
    Jz2 = matrix_power(Jz, 2)
    Jz3 = matrix_power(Jz, 3)
    Jz4 = matrix_power(Jz, 4)
    Jz5 = matrix_power(Jz, 5)
    Jz6 = matrix_power(Jz, 6)

    Jadd2 = matrix_power(Jadd, 2)
    Jadd3 = matrix_power(Jadd, 3)
    Jadd4 = matrix_power(Jadd, 4)
    Jadd5 = matrix_power(Jadd, 5)
    Jadd6 = matrix_power(Jadd, 6)

    Jminus2 = matrix_power(Jminus, 2)
    Jminus3 = matrix_power(Jminus, 3)
    Jminus4 = matrix_power(Jminus, 4)
    Jminus5 = matrix_power(Jminus, 5)
    Jminus6 = matrix_power(Jminus, 6)

    I = np.identity(njs)
    X = J*(J + 1)*I
    X2 = matrix_power(X, 2)
    X3 = matrix_power(X, 3)
    
    # define Stevens operators (including order 0, 2, 4, 6)
    # Order 0
    O00 = I

    # Order 2
    O20 = 3*Jz2 - X
    O21 = 1/4*(Jz@(Jadd + Jminus) + (Jadd + Jminus)@Jz)
    O21s = -1j/4*(Jz@(Jadd - Jminus) + (Jadd - Jminus)@Jz)
    O22 = 1/2*(Jadd2 + Jminus2)
    O22s = -1j/2*(Jadd2 - Jminus2)

    # Order 4
    O40 = 35*Jz4 - (30*X - 25*I)@Jz2 + 3*X2 - 6*X
    
    O41_part = (7*Jz3 - (3*X + I)@Jz)
    O41 = 1/4*((Jadd + Jminus)@O41_part + O41_part@(Jadd + Jminus))
    O41s = -1j/4*((Jadd - Jminus)@O41_part + O41_part@(Jadd - Jminus))
    
    O42_part = 7*Jz2 - X - 5*I
    O42 = 1/4*((Jadd2 + Jminus2)@O42_part + O42_part@(Jadd2 + Jminus2))
    O42s = -1j/4*((Jadd2 - Jminus2)@O42_part + O42_part@(Jadd2 - Jminus2))
    
    O43 = 1/4*((Jadd3 + Jminus3)@Jz + Jz@(Jadd3 + Jminus3))
    O43s = -1j/4*((Jadd3 - Jminus3)@Jz + Jz@(Jadd3 - Jminus3))
    
    O44 = 1/2*(Jadd4 + Jminus4)
    O44s = -1j/2*(Jadd4 - Jminus4)

    
    # Order 6    
    O60 = (231*Jz6 -
           (315*X - 735*I)@Jz4 +
           (105*X2 - 525*X + 294*I)@Jz2 -
           5*X3 + 40*X2 - 60*X)
    
    O61_part = 33*Jz5 - (30*X - 15*I)@Jz3 + (5*X2 - 10*X + 12*I)@Jz
    O61 = 1/4*((Jadd + Jminus)@O61_part + O61_part@(Jadd + Jminus))
    O61s = -1j/4*((Jadd - Jminus)@O61_part + O61_part@(Jadd - Jminus))
    
    O62_part = 33*Jz4 - (18*X + 123*I)@Jz2 + X2 + 10*X + 102*I
    O62 = 1/4*((Jadd2 + Jminus2)@O62_part + O62_part@(Jadd2 + Jminus2))
    O62s = -1j/4*((Jadd2 - Jminus2)@O62_part + O62_part@(Jadd2 - Jminus2))
    
    O63_part = 11*Jz3 - (3*X + 59*I)@Jz
    O63 = 1/4*((Jadd3 + Jminus3)@O63_part + O63_part@(Jadd3 + Jminus3))
    O63s = -1j/4*((Jadd3 - Jminus3)@O63_part + O63_part@(Jadd3 - Jminus3))

    O64_part = 11*Jz2 -X - 38*I
    O64 = 1/4*((Jadd4 + Jminus4)@O64_part + O64_part@(Jadd4 + Jminus4))
    O64s = -1j/4*((Jadd4 - Jminus4)@O64_part + O64_part@(Jadd4 - Jminus4))

    O65 = 1/4*((Jadd5 + Jminus5)@Jz + Jz@(Jadd5 + Jminus5))
    O65s = -1j/4*((Jadd5 - Jminus5)@Jz + Jz@(Jadd5 - Jminus5))
    
    O66 = 1/2*(Jadd6 + Jminus6)
    O66s = -1j/2*(Jadd6 - Jminus6)
    
    # store all the Stevens operators in a dictionary 
    O_dict = {}
    O_dict[0] = {0: O00}
    O_dict[2] = {-2: O22s, -1: O21s, 0: O20, 1: O21, 2: O22,}
    O_dict[4] = {-4: O44s, -3: O43s, -2: O42s, -1: O41s, 0: O40, 1: O41, 2: O42, 3: O43, 4: O44}
    O_dict[6] = {-6: O66s, -5: O65s, -4: O64s, -3: O63s, -2: O62s, -1: O61s, 0: O60, 
                 1: O61, 2: O62, 3: O63, 4: O64, 5: O65, 6: O66}

    # calculate the crystal field matrix based on the given Stevens parameters. See the first equation in https://www2.cpfs.mpg.de/~rotter/homepage_mcphase/manual/node132.html
    cfstevens = np.zeros((njs, njs), dtype=np.complex128)
    for k in B:
        for m in B[k]:
            cfstevens += B[k][m] * O_dict[k][m]
    return cfstevens

def J2LS(ion, B):
    """
    Convert the Stevens parameters from (J, Jz) basis to LS basis.
    
    Parameters
    ----------
    ion : string
          Name of the rare earth ion.
    B : dictionary
        Conventional Stevens parameters.

    Returns
    -------
    B : dictionary
        New stevens parameters after the conversion.

    Notes
    -----
    The reason for this conversion is because Stevens parameters contain a multiplicative factor $\theta_k = <J|theta_k|J>$,
    which depends on J, and the conventional Stevens parameters are defined only for the lowest energy multiplet J. 
    So we have to use the correct $\theta_k$ for the J value (which is the L value in the LS basis in our case) we choose.
        
    # "Thet" contains the original $\theta_k$ values from Hutchings' Table VI (Hutchings, Solid State Physics - Advances in 
    # Research and Applications 16, 227 (1964)).
    # the order is k=2, k=4, k=6 for each entry.
    """
    Thet = {}
    Thet['Ce3+'] = [-2./(5*7), 2./(3*3*5*7), 0]
    Thet['Pr3+'] = [-2.*2*13/(3*3*5*5*11), -2.*2/(3*3*5*11*11), 2.**4*17/(3**4*5*7*11**2*13)]
    Thet['Nd3+'] = [-7./(3**2*11**2) , -2.**3*17/(3**3*11**3*13), -5.*17*19/(3**3*7*11**3*13**2)]
    Thet['Pm3+'] = [2*7./(3*5*11**2), 2.**3*7*17/(3**3*5*11**3*13), 2.**3*17*19/(3**3*7*11**2*13**2)]
    Thet['Sm3+'] = [13./(3**2*5*7) , 2.*13/(3**3*5*7*11), 0]
    Thet['Tb3+'] = [-1./(3**2*11), 2./(3**3*5*11**2), -1./(3**4*7*11**2*13)]
    Thet['Dy3+'] = [-2./(3**2*5*7) , -2.**3/(3**3*5*7*11*13), 2.*2/(3**3*7*11**2*13**2)]
    Thet['Ho3+'] = [-1./(2*3*3*5*5), -1./(2*3*5*7*11*13), -5./(3**3*7*11**2*13**2)]
    Thet['Er3+'] = [2.*2/(3*3*5*5*7) , 2./(3.**2*5*7*11*13), 2.*2*2/(3.**3*7*11**2*13**2)]
    Thet['Tm3+'] = [1./(3**2*11) , 2.**3/(3**4*5*11**2), -5./(3**4*7*11**2*13)]
    Thet['Yb3+'] = [2./(3**2*7) , -2./(3*5*7*11), 2.*2/(3**3*7*11*13)]
    def theta(ion,k):
        return Thet[ion][int(k/2-1)]
    
    # "LSThet" contains the $\theta_k$ values for LS basis. Find more details about the calculation in A. Scheie, PyCrystalField: 
    # Software for Calculation, Analysis and Fitting of Crystal Electric Field Hamiltonians, Journal of Applied Crystallography 54, 356 (2021).
    # These values are taken from PyCrystalField (https://github.com/asche1/PyCrystalField/blob/master/src/pcf_lib/PointChargeConstants.py)
    # the order is k=2, k=4, k=6 for each entry.
    LSThet = {}
    LSThet['Sm3+'] = [0.0148148148148, 0.0003848003848, -2.46666913334e-05]
    LSThet['Pm3+'] = [0.0040404040404, 0.000122436486073, 1.12121324243e-05]
    LSThet['Nd3+'] = [-0.0040404040404, -0.000122436486073, -1.12121324243e-05]
    LSThet['Ce3+'] = [-0.0444444444444, 0.0040404040404, -0.001036001036]
    LSThet['Dy3+'] = [-0.0148148148148, -0.0003848003848, 2.46666913334e-05]
    LSThet['Ho3+'] = [-0.0040404040404, -0.000122436486073, -1.12121324243e-05]
    LSThet['Tm3+'] = [0.0148148148148, 0.0003848003848, -2.46666913334e-05]
    LSThet['Pr3+'] = [-0.0148148148148, -0.0003848003848, 2.46666913334e-05]
    LSThet['Er3+'] = [0.0040404040404, 0.000122436486073, 1.12121324243e-05]
    LSThet['Tb3+'] = [-0.0444444444444, 0.0040404040404, -0.001036001036]
    LSThet['Yb3+'] = [0.0444444444444, -0.0040404040404, 0.001036001036]
    def LStheta(ion,k):
        return LSThet[ion][int(k/2-1)]
    
    # do the conversion
    new_B = {}
    for k in B:
        new_B[k] = {}
        if k == 0: # the B00 term does not matter, so we simply neglect this term in the conversion
            new_B[0][0] = B[0][0]
        else:
            for m in B[k]:
                new_B[k][m] = B[k][m] / theta(ion, k) * LStheta(ion,k)
    return new_B