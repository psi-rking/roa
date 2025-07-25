"""
  scatter() function for ROA spectra

  Accept three tensors needed for ROA.  Compute
    their derivatives along normal modes, then compute ROA parameters.
    ROA requires the following polarizability tensors:
      (1) electric-dipole/electric-dipole;      A = polarizability
      (2) electric-dipole/magnetic-dipole; and  G = opt. rot. tensor
      (3) electric-dipole/electric-quadrupole.  Q = dipole/quad. polar.

  Nomenclature of variables:
       displaced  derivative derivative  derivative
        tensors    wrt xyz    wrt xyz     wrq q (norm mode)
    (1)  A_fd      A_grad      A_der      A_der_q
    (2)  G_fd      G_grad      G_der      G_der_q
    (3)  Q_fd      Q_grad      Q_der      Q_der_q
    dipole derivatives        mu_der     mu_der_q
     _der is the same as _grad - just organized in one big supermatrix.

  Original version in ccresponse in C++ by TDC, August 2009
  Adapted and extended in Python by RAK, summer 2019
"""
import numpy as np
from math import pi, sqrt, acos
from qcelemental import constants
#pc_dipmom_debye2si = constants.dipmom_debye2si
#pc_e0              = constants.e0
pc_au2amu          = constants.au2amu
pc_amu2kg          = constants.amu2kg
pc_c               = constants.c
pc_na              = constants.get("Avogadro constant")
pc_a0              = constants.get("atomic unit of length")
pc_bohr2angstroms  = constants.bohr2angstroms
pc_hartree2J       = constants.hartree2J
pc_me              = constants.me
pc_h               = constants.h
from psi4.core import print_out, BasisSet
from psi4.driver import qcdb

# 'Energy' in wavenumbers = 1/lambda = E/hc from freq. of harmonic oscillator
#constants_HOfreq2wn = 1.0 / (2.0 * pi * pc_c * 100.0)

# constants_c_au is the speed of light in au = 1/alpha = 137.0 ...
constants_c_au = pc_c * pc_me * pc_bohr2angstroms * 1e-10 / (pc_h / (2.0 * pi))

# from square of the dipole strength to IR intensity in km/mol
constants_dipole_to_kmmol = (pc_na * np.pi * 1.e-3 * constants.get("electron mass in u") *
                   constants.get("fine-structure constant")**2 * pc_a0 / (3*pc_au2amu))

# For each vibrational mode, convert Raman scattering parameters
# from a0^4 / au -> Ang^4 / amu.
constants_raman_conv = pc_bohr2angstroms * pc_bohr2angstroms * pc_bohr2angstroms * \
             pc_bohr2angstroms / pc_au2amu


def modeVectorsQCDB(mol, hessian, modesToReturn=None, print_lvl=1, pr=print,
                    printMolden=False):
    Natom = mol.natom()
    geom = mol.geometry().to_array()
    masses = np.asarray([mol.mass(at) for at in range(Natom)]) # masses in amu
    atom_symbols = [mol.symbol(at) for at in range(Natom)] 

    pr("\n\tCalculating normal modes; Input coordinates (bohr)\n")
    for i in range(Natom):
        pr("\t{:5s}{:10.5f}{:15.10f}{:15.10f}{:15.10f}\n".format(
           atom_symbols[i],masses[i],geom[i,0],geom[i,1],geom[i,2]))

    basis = BasisSet.build(mol) # dummy bs to lookup symmetry info
    irrep_lbls = mol.irrep_labels() # labels of IRREPs if molecule has symmetry
    dipder = None  # unneeded dipole-moment derivatives unless we add intensities

    vibinfo, vibtext = qcdb.vib.harmonic_analysis(hessian, geom, masses, basis,
                        irrep_lbls, dipder, project_trans=True, project_rot=True)

    if printMolden:
      s = qcdb.vib.print_molden_vibs(vibinfo, atom_symbols, geom, standalone=True)
      with open('molden.vibs','w') as handle:
          handle.write(s)

    freqs = vibinfo['omega'].data.real
    # modes from vibinfo are unmass-weighted, not normalized, returned as cols
    # in amu^-1/2 ; convert here to au^-1/2
    modes = sqrt(pc_au2amu) * vibinfo['w'].data.T

    # Resort from high to low freqs
    modes[:] = np.flip(modes, axis=0)
    freqs[:] = np.flip(freqs)

    pr("\n\t----------------------\n")
    pr("\t Mode #   Harmonic      \n")
    pr("\t          Freq. (cm-1)  \n")
    pr("\t------------------------\n")
    for i in range(0,3*Natom):
        if freqs[i] < 0.0:
            pr("\t  %3d   %10.2fi\n" % (i, abs(freqs[i])))
        else:
            pr("\t  %3d   %10.2f \n" % (i, freqs[i]))

    selected_modes = np.zeros( (len(modesToReturn),3*Natom))
    selected_freqs = np.zeros( len(modesToReturn) )

    for i, mode in enumerate(modesToReturn):
        selected_modes[i,:] = modes[mode,:]
        selected_freqs[i] = freqs[mode]

    pr("\nFrequencies (with modes) returned from modeVectorsPsi4.\n")
    pr(str(selected_freqs)+'\n')

    if print_lvl > 1:
        pr("\nVectors returned from modeVectorsPsi4 are rows.\n")
        pr(str(selected_modes)+'\n')

    return (selected_modes, selected_freqs)


"""
  modeScatter() function for ROA spectra
  See heading of scatter() function.
  This function does the same ROA analysis but only on a given
  set of normal modes.  It does not do the Hessian analysis.
"""
def modeScatter(
        mode_indices,# list   : indices of mode used as label.
        mode_vectors,# 2Darray: vectors in mode direction (Lx)
        mode_freqs,  # array  : harmonic frequency of mode
        geom,       # Cartesian coordinates of nuclei
        masses,     # atomic masses
        mu_der,     # Cartesian dipole moment derivatives
        omega,      # energy in au (caller can use omega_in_au to convert if needed)
        step,       # displacement-size for finite differences
        A_fd,       # Lists of 3-point finite-difference ndarrays
        G_fd,       #   input order alternates + and - for each Cartesian
        Q_fd,       # 
        print_lvl=1,# 1==minimal
        ROAdictName='spectra.dat', # name of dictionary output file
        calc_type='Calc Type',  # for output, if desired
        nbf=None,
        pr=print,
        modes2decompose=[1]
      ):

    Natom = len(geom)
    Nmodes = len(mode_indices)
    pr("\nInput coordinates (bohr)\n")
    for a in range(Natom):
        pr("%15.10f%15.10f%15.10f\n" % (geom[a,0],geom[a,1],geom[a,2]))

    # COMPUTE TENSOR DERIVATIVES
    # August 2023: Adding negative sign to finite-differences so these
    # are technically derivatives, not gradients.  This has no effect on
    # the products of derivatives that appear in the VROA parameters.
    #
    # A_grad = Array of length # of Cartesian coordinates with respect
    # to which derivative is taken.  Each member is a 3x3 ndarray for the
    #  various electric-dipole/electric-dipole polarizability components.
    A_grad = []
    for i in range(0, len(A_fd), 2):
        grad_mat = -np.subtract( A_fd[i], A_fd[i+1] )
        grad_mat[:] /=  (2.0 * step)
        A_grad.append(grad_mat)

    # G_grad = Array of length # of Cartesian coordinates with respect
    # to which derivative is taken.  Each member is a 3x3 ndarray for the
    #  various electric-dipole/magnetic-dipole polarizability components.
    G_grad = []
    for i in range(0, len(G_fd), 2):
        grad_mat = -np.subtract( G_fd[i], G_fd[i+1] )
        grad_mat[:] /=  (2.0 * step)
        G_grad.append(grad_mat)

    # Q_grad = Array of length # of Cartesian coordinates with respect
    # to which derivative is taken.  Each member is a 9x3 ndarray for the
    #  various electric quadropole/electric-dipole polarizability components.
    Q_grad = []
    for i in range(0, len(G_fd), 2):
        grad_mat = -np.subtract( Q_fd[i], Q_fd[i+1] )
        grad_mat[:] /=  (2.0 * step)
        Q_grad.append(grad_mat)

    if print_lvl >= 2:
        s =  "\n******************************************************\n"
        s += "******   ROA TENSOR DERIVATIVES wrt given modes  *****\n"
        s += "******************************************************\n\n"
        s += "\t*** Dipole Polarizability Derivative Tensors ***\n\n"
        for i in range(len(A_grad)):
            s += A_grad[i].__str__() + '\n'
        s += "\n"
        s += "*********************************************************\n\n"
        s += "\t*** Optical Rotation Derivative Tensors ***\n\n"
        for i in range(len(G_grad)):
            s += G_grad[i].__str__() + '\n'
        s += "\n"
        s += "*********************************************************\n\n"
        s += "\t*** Dipole/Quadrupole Derivative Tensors ***\n\n"
        for i in range(len(Q_grad)):
            s += Q_grad[i].__str__() + '\n'
        s += "\n"
        pr(s)

    # Pull each tensor array into a single supermatrix
    A_der_q_tmp = np.zeros( (Nmodes,9) )
    G_der_q_tmp = np.zeros( (Nmodes,9) )
    Q_der_q_tmp = np.zeros( (Nmodes,27) )
    for i in range(Nmodes):
        A_der_q_tmp[i,:] = A_grad[i].reshape(1,9)
        G_der_q_tmp[i,:] = G_grad[i].reshape(1,9)
        Q_der_q_tmp[i,:] = Q_grad[i].reshape(1,27)

    Lx = mode_vectors.T # after transpose, columns are displacement vectors

    redmass = np.zeros( Nmodes )
    for i in range(Nmodes):
        norm = 0.0;
        for j in range(3*Natom):
            norm += Lx[j,i] * Lx[j,i] / pc_au2amu
        if norm > 1e-3:
            redmass[i] = 1.0 / norm

    # Transform dipole-moment derivatives to normal coordinates
    # mu_der   is 3 x 3*Natom
    # Lx       is 3*Natom * Nmodes
    # mu_der_q is 3 x Nmodes
    mu_der_q = np.dot(mu_der, Lx)

    # Compute IR intensities in projected normal coordinates
    #mu_der_conv = pc_dipmom_debye2si * pc_dipmom_debye2si / (1e-20 * pc_amu2kg * pc_au2amu) \
    #              * pc_na * pi / (3.0 * pc_c * pc_c * 4.0 * pi * pc_e0 * 1000.0)

    IRint = np.zeros(Nmodes)
    for i in range(Nmodes):
        for j in range(3): # include au -> amu in constants_dipole_to_kmmol
            IRint[i] += constants_dipole_to_kmmol * mu_der_q[j,i] * mu_der_q[j,i]

    # Transform polarizability derivatives to normal coordinates
    # A_der   is 3*Natom x 9
    # Lx      is 3*Natom x 3*Natom
    # A_der_q is 9 x 3*Natom
    #A_der_q_tmp = np.dot(A_der.T, Lx)
    #if print_lvl >= 2:
    #    pr("\tPolarizability Derivatives in Normal Coord.\n")
    #    pr(str(A_der_q_tmp)+'\n')

    # Reorganize list of 3x3 polarizability derivatives for each Cartesian coordinate
    A_der_q = []
    for i in range(Nmodes):
        one_alpha_der = np.zeros( (3,3) )
        jk = 0
        for j in range(3):
            for k in range(3):
                one_alpha_der[j,k] = A_der_q_tmp[i,jk]
                jk += 1
        A_der_q.append(one_alpha_der)

    # Transform optical rotation tensor derivatives to normal coordinates
    # G_der   is 3*Natom x 9
    # Lx      is 3*Natom x 3*Natom
    # G_der_q is 9 x 3*Natom
    #G_der_q_tmp = np.dot(G_der.T, Lx)
    #if print_lvl >= 2:
    #    pr("\n\tOptical Rotation Tensor Derivatives in Normal Coord.\n")
    #    pr(str(G_der_q_tmp)+'\n')

    # Reorganize list of optical rotation derivatives for each Cartesian coordinate
    G_der_q = []
    for i in range(Nmodes):
        one_G_der_q = np.zeros( (3,3) )
        jk = 0
        for j in range(3):
            for k in range(3):
                one_G_der_q[j,k] = G_der_q_tmp[i,jk]
                jk += 1
        G_der_q.append(one_G_der_q)

    # Transform dipole/quadrupole tensor derivatives to normal coordinates
    #Q_der_q_tmp = np.dot(Q_der.T, Lx)
    #if print_lvl >= 2:
    #    pr("Dipole/Quadrupole Tensor Derivatives in Normal Coord.\n")
    #    pr(str(Q_der_q_tmp)+'\n')

    Q_der_q = np.zeros( (Nmodes,3,3,3) )
    for i in range(Nmodes):
        jkl = 0
        for j in range(3):
            for k in range(3):
                for l in range(3):
                    Q_der_q[i][j][k][l] = Q_der_q_tmp[i,jkl]
                    jkl += 1

    # This is confusing, but to try to fit with the literature from this point on 
    # alpha = polarizability; A = quadrupole term !
    freqs           = mode_freqs
    alpha           = np.zeros(Nmodes)
    beta_alpha2     = np.zeros(Nmodes)
    ramint_linear   = np.zeros(Nmodes)
    depol_linear    = np.zeros(Nmodes)
    ramint_circular = np.zeros(Nmodes)
    depol_circular  = np.zeros(Nmodes)

    for i in range(Nmodes):
        # isotropic invariant of electric-dipole polarizability
        alpha[i]           = f_tensor_mean(A_der_q[i])
        # anisotropic invariant for electric dipole polarizability
        beta_alpha2[i]     = f_tensor_zeta(A_der_q[i], A_der_q[i])
        ramint_linear[i]   = f_raman_linear(alpha[i], beta_alpha2[i])
        depol_linear[i]    = f_depolar_linear(alpha[i], beta_alpha2[i])
        ramint_circular[i] = f_raman_circular(alpha[i], beta_alpha2[i])
        depol_circular[i]  = f_depolar_circular(alpha[i], beta_alpha2[i])

    # compute the frequencies and spit them out in a nice table in the output
    pr("\nThis stuff only depends on polarizability and dipole moment derivatives.\n")
    pr("\tAngular frequency (in au): = %15.10f\n" % omega)
    pr("--------------------------------------------------------------------------------------------------------\n")
    pr("                              Raman Scattering Parameters\n")
    pr("--------------------------------------------------------------------------------------------------------\n")
    pr("     Harmonic      IR        Red.     alpha^2   Beta(     |   Raman       Dep.  | Raman        Dep.\n")
    pr("       Freq.    Intensity    Mass                alpha)^2 |    Act.      Ratio  |  Act.       Ratio\n")
    pr("      (cm-1)    (km/mol)     (amu)                        |      (linear)       |    (circular)  \n")
    pr("--------------------------------------------------------------------------------------------------------\n")
    for i in range(0, Nmodes):
        if freqs[i] < 0.0:
            pr("%3d %8.2fi %9.4f    %7.4f %9.4f   %9.4f  %9.4f  %9.4f  %9.4f  %9.4f\n" % (i+1,
                            abs(freqs[i]), IRint[i], redmass[i], alpha[i] * alpha[i] * constants_raman_conv,
                            beta_alpha2[i] * constants_raman_conv, ramint_linear[i] * constants_raman_conv, depol_linear[i],
                            ramint_circular[i] * constants_raman_conv, depol_circular[i]))
        else:
            pr( "%3d %9.2f %9.4f    %7.4f  %9.4f  %9.4f  %9.4f  %9.4f  %9.4f  %9.4f\n" % (i+1,
                            freqs[i], IRint[i], redmass[i], alpha[i] * alpha[i] * constants_raman_conv,
                            beta_alpha2[i] * constants_raman_conv, ramint_linear[i] * constants_raman_conv, depol_linear[i],
                            ramint_circular[i] * constants_raman_conv, depol_circular[i]) )
    pr("--------------------------------------------------------------------------------------------------------\n")


    G               = np.zeros(Nmodes)
    beta_G2         = np.zeros(Nmodes)
    betaA2          = np.zeros(Nmodes)
    robustnessPhi   = np.zeros(Nmodes)
    robustnessPsi   = np.zeros(Nmodes)

    for i in range(Nmodes):
        # isotropic invariant of electric-dipole/magnetic dipole optical activity tensor
        G[i]               = f_tensor_mean(G_der_q[i])
        # anisotropic invariant for electric-dipole/magnetic dipole polarizability
        beta_G2[i]         = f_tensor_zeta(A_der_q[i], G_der_q[i])

        robustnessPhi[i]   = f_robustnessPhi(A_der_q[i], G_der_q[i])
        robustnessPsi[i]   = f_robustnessPsi(A_der_q[i], Q_der_q[i])
        betaA2[i] = f_beta_A2(A_der_q[i], Q_der_q[i], omega)

    roa_conv = constants_raman_conv * 1e6 / constants_c_au;
    pr("----------------------------------------------------------------------\n")
    pr("               ROA Scattering Invariants\n")
    pr("----------------------------------------------------------------------\n")
    pr("       Frequency    alpha*G    Beta(G)^2   Beta(A)^2    phi    psi\n")
    pr("----------------------------------------------------------------------\n")
    #for i in range(Nmodes-1, -1, -1):
    for i in range(0, Nmodes):
        if freqs[i] < 0.0:
            pr("  %3d  %9.3fi %9.4f   %10.4f  %10.4f %7.1f %7.1f\n"  % (i+1,
                            abs(freqs[i]), alpha[i] * G[i] * roa_conv,
                            beta_G2[i] * roa_conv, betaA2[i] * roa_conv, robustnessPhi[i], robustnessPsi[i]))
        else:
            pr("  %3d  %9.3f  %9.4f   %10.4f  %10.4f %7.1f %7.1f\n" %  (i+1,
                            freqs[i], alpha[i] * G[i] * roa_conv,
                            beta_G2[i] * roa_conv, betaA2[i] * roa_conv, robustnessPhi[i], robustnessPsi[i]))
    pr("---------------------------------------------------------------------\n")

    pr("----------------------------------------------------------------------\n")
    pr("         ROA Difference Parameter R-L (Angstrom^4/amu * 1000)\n")
    pr("----------------------------------------------------------------------\n")
    pr("     Harmonic Freq.  Delta_z      Delta_x       Delta        Delta \n")
    pr("       (cm^-1)         (90)         (90)          (0)        (180)  \n")
    pr("----------------------------------------------------------------------\n")

    delta_0   = np.zeros(Nmodes)
    delta_180 = np.zeros(Nmodes)
    delta_x   = np.zeros(Nmodes)
    delta_z   = np.zeros(Nmodes)

    for i in range(Nmodes):
        delta_0[i]   = constants_raman_conv * 1e3 * 4.0 * (
          180.0 * alpha[i] * G[i] + 4.0 * beta_G2[i] - 4.0 * betaA2[i]) / constants_c_au
        delta_180[i] = constants_raman_conv * 1e3 * 4.0 * (
          24.0 * beta_G2[i] + 8.0 * betaA2[i]) / constants_c_au
        delta_x[i]   = constants_raman_conv * 1e3 * 4.0 * (
          45.0 * alpha[i] * G[i] + 7.0 * beta_G2[i] + betaA2[i]) / constants_c_au
        delta_z[i]   = constants_raman_conv * 1e3 * 4.0 * (
           6.0 * beta_G2[i] - 2.0 * betaA2[i]) / constants_c_au

    #for i in range(Nmodes-1, -1, -1):
    for i in range(0, Nmodes):
        if freqs[i] < 0.0:
            pr("  %3d  %9.2fi %10.4f   %10.4f   %10.4f   %10.4f\n" % (i+1,
                 abs(freqs[i]), delta_z[i], delta_x[i], delta_0[i], delta_180[i]))
        else:
            pr("  %3d  %9.2f  %10.4f   %10.4f   %10.4f   %10.4f\n" % (i+1,
                 freqs[i], delta_z[i], delta_x[i], delta_0[i], delta_180[i]))
    pr("----------------------------------------------------------------------\n")

    Dout = {}
    Dout['Calculation Type']           = calc_type
    if nbf is not None:
        Dout['Number of basis functions']  = nbf
    Dout['Frequency']                  = freqs[0:Nmodes]
    Dout['IR Intensity']               = IRint[0:Nmodes]
    Dout['Raman Intensity (linear)']   = constants_raman_conv * ramint_linear[0:Nmodes]
    Dout['Raman Intensity (circular)'] = constants_raman_conv * ramint_circular[0:Nmodes]
    Dout['ROA R-L Delta(0)']           = delta_0[0:Nmodes]
    Dout['ROA R-L Delta(180)']         = delta_180[0:Nmodes]
    Dout['ROA R-L Delta(90)_x']        = delta_x[0:Nmodes]
    Dout['ROA R-L Delta(90)_z']        = delta_z[0:Nmodes]
    Dout['ROA alpha*G']                = alpha[0:Nmodes] * G[0:Nmodes]
    Dout['ROA Beta(G)^2']              = beta_G2[0:Nmodes]
    Dout['ROA Beta(A)^2']              = betaA2[0:Nmodes]
    with open(ROAdictName, "w") as f:
        f.write(str(Dout))

    for m in modes2decompose:
        local_pairs(A_der_q, G_der_q, Q_der_q, Lx, omega, roa_conv, m-1, freqs, pr)

    return


# The Levi-Civitas evaluator
def levi(a, b, c):
    x,y,z = 0,1,2

    if   (a,b,c) == (x, y, z): val =  1
    elif (a,b,c) == (y, z, x): val =  1
    elif (a,b,c) == (z, x, y): val =  1
    elif (a,b,c) == (x, z, y): val = -1
    elif (a,b,c) == (y, x, z): val = -1
    elif (a,b,c) == (z, y, x): val = -1
    else: val = 0

    return val

# Compute isotropic mean of a property tensor
def f_tensor_mean(X):
    mean = np.sum(X[i,i] for i in range(3)) / 3.0
    return mean

# Compute anisotropic invariant of a property tensor 
# beta(X)^2  = 1/2 [3 * X_ij*X_ij      - X_ii*X_jj.
# beta(G')^2 = 1/2 [3 * alpha_ij*G'_ij - alpha_ii*G'_jj
def f_tensor_zeta(X, Y):
    value = 0.0
    for i in range(3):
        for j in range(3):
            value += 0.5 * (3.0 * X[i, j] * Y[i, j] - X[i, i] * Y[j, j])
    return value

# Compute beta(A)^2 = 1/2 omega * alpha_ij epsilon_ikl * A_klj
def f_beta_A2(alpha, A, omega):
    value = 0.0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                for l in range(3): 
                    value += 0.5 * omega * alpha[i, j] * levi(i, k, l) * A[k][l][j]
    return value

# Compute Raman intensity for linearly polarized radiation.
#    A = 45 (alpha^2) + 4 (beta^2)
def f_raman_linear(alpha, beta2):
    value = 45.0 * alpha * alpha + 4.0 * beta2
    return value

# Compute Raman depolarization ratio for 90-degree scattering of
# linearly polarized radiation:
#  ratio = [ 3 * beta(alpha)^2)/(45 * alpha^2 + 4 * beta(alpha)^2) ]
def f_depolar_linear(alpha, beta2):
    numer = 3.0 * beta2
    denom = (45.0 * alpha * alpha) + (4.0 * beta2)
    ratio = 0.0
    if denom > 1e-6:
        ratio =  numer / denom
    return ratio

# Compute Raman intensity for circularly polarized radiation:
#  A = 45 (alpha^2) + 7 (beta^2)
def f_raman_circular(alpha, beta2):
    value = 45.0 * alpha * alpha + 7.0 * beta2
    return value

# Compute Raman depolarization ratio for 90-degree scattering of
# circularly polarized radiation:
#  ratio = [ 6 * beta(alpha)^2)/(45 * alpha^2 + 7 * beta(alpha)^2) ] //
def f_depolar_circular(alpha, beta2):
    numer = 6.0 * beta2
    denom = (45.0 * alpha * alpha) + (7.0 * beta2)
    ratio = 0.0
    if denom > 1e-6:
        ratio = numer / denom
    return ratio

# Lq has dimensions (3N, Nmodes)
# A_der_q is (Nmodes, 3, 3)
def local_pairs(A_der_q, G_der_q, Q_der_q, Lq, omega, roa_conv, mode, wavenum, pr):
    Natom = Lq.shape[0]//3
    Nmodes = Lq.shape[1]
    #pr('\nLq.shape:' + str(Lq.shape))
    #pr('\nNatom: %d' % Natom)
    #pr('\nNmodes: %d' % Nmodes)
    #pr('\nA_der_q[mode].shape: %s\n' % str(A_der_q[mode].shape))

    # Find generalized left-inverse so Linv Lq = I (Nmodes,Nmodes)
    Linv = np.linalg.pinv(Lq)   # Linv has dimensions (Nmodes, 3N)
    #pr('Linv:' + str(Linv) + '\n')

    # This is a check; gives the correct Beta(A2) total.
    #total = 0.0
    #for i in range(3): 
    #    for j in range(3): 
    #        for k in range(3):
    #            for l in range(3):
    #                total += 0.5 * omega * A_der_q[mode][i,j] * levi(i,k,l) * Q_der_q[mode][k,l,j]
    #pr('Sum betaA2: %15.10f\n' % (roa_conv * total))

    # Need to determine the derivative wrt xyz of each atoms separately - before
    # contracting them together, so this is roundabout.
    A_der_q_xyz = np.zeros( (3*Natom,3,3) )
    for a in range(3*Natom):
      for i in range(3):
        for j in range(3):
          A_der_q_xyz[a,i,j] = Linv[mode,a] * A_der_q[mode][i,j]

    G_der_q_xyz = np.zeros( (3*Natom,3,3) )
    for a in range(3*Natom):
      for i in range(3):
        for j in range(3):
          G_der_q_xyz[a,i,j] = Linv[mode,a] * G_der_q[mode][i,j]

    Q_der_q_xyz = np.zeros( (3*Natom, 3,3,3) )
    for a in range(3*Natom):
      for k in range(3):
        for l in range(3):
          for j in range(3):
            Q_der_q_xyz[a,k,l,j] = Linv[mode,a] * Q_der_q[mode][k,l,j]

    # Now calculate Cartesian pairwise contributions to each tensor.
    # beta(A2) = 1/2 omega d(alpha_ij)/da levi_ikl d(Q_klj)/db
    beta_A2_local_xyz_mat = np.zeros( (3*Natom, 3*Natom) )
    for Ax in range(3*Natom):
        for Bx in range(3*Natom):
            value = 0.0 
            for i in range(3): 
                for j in range(3): 
                    for k in range(3):
                        for l in range(3):
                            value += 0.5 * omega * A_der_q_xyz[Ax,i,j] * levi(i,k,l) * Q_der_q_xyz[Bx,k,l,j]
            beta_A2_local_xyz_mat[Ax][Bx] = value

    # Compute beta(G')^2 = 1/2[3 * d(alpha_ij)/da * d(G'_ij)/db - d(alpha_ii)da  * d(G'_jj)/db
    beta_G2_local_xyz_mat = np.zeros( (3*Natom, 3*Natom) )
    for Ax in range(3*Natom):
        for Bx in range(3*Natom):
            value = 0.0
            for i in range(3):
                for j in range(3):
                    value += 0.5 * (3.0 * A_der_q_xyz[Ax,i,j] * G_der_q_xyz[Bx,i,j] -
                                          A_der_q_xyz[Ax,i,i] * G_der_q_xyz[Bx,j,j])
            beta_G2_local_xyz_mat[Ax][Bx] = value

    # Compute alpha*G' = d(alpha_i)/da * d(G'_i)/db 
    alphaG_local_xyz_mat = np.zeros( (3*Natom, 3*Natom) )
    for Ax in range(3*Natom):
        for Bx in range(3*Natom):
            value = 0.0
            for i in range(3):
                for j in range(3):
                    value += A_der_q_xyz[Ax,i,i] * G_der_q_xyz[Bx,j, j] / 9.0
            alphaG_local_xyz_mat[Ax][Bx] = value

    # Now add up each atomic pairwise contribution to the sum for the normal
    # mode.  Symmetrize and divide diagonal by 2.0 for fair counting.
    beta_A2_local_pairwise = np.zeros( (Natom,Natom) )
    for A in range(Natom):
        for B in range(Natom):
            val = 0
            for Ax in range(3*A, 3*A+3):
                for Bx in range(3*B, 3*B+3):
                    val += Lq[Ax,mode] * beta_A2_local_xyz_mat[Ax,Bx] * Lq[Bx,mode]

            beta_A2_local_pairwise[A,B] += val

    beta_A2_local_pairwise[:] = np.add(beta_A2_local_pairwise, beta_A2_local_pairwise.T)
    for A in range(Natom):
        beta_A2_local_pairwise[A,A] /= 2.0

    beta_G2_local_pairwise = np.zeros( (Natom,Natom) )
    for A in range(Natom):
        for B in range(Natom):
            val = 0
            for Ax in range(3*A, 3*A+3):
                for Bx in range(3*B, 3*B+3):
                    val += Lq[Ax,mode] * beta_G2_local_xyz_mat[Ax,Bx] * Lq[Bx,mode]

            beta_G2_local_pairwise[A,B] += val

    beta_G2_local_pairwise[:] = np.add(beta_G2_local_pairwise, beta_G2_local_pairwise.T)
    for A in range(Natom):
        beta_G2_local_pairwise[A,A] /= 2.0

    alphaG_local_pairwise = np.zeros( (Natom,Natom) )
    for A in range(Natom):
        for B in range(Natom):
            val = 0
            for Ax in range(3*A, 3*A+3):
                for Bx in range(3*B, 3*B+3):
                    val += Lq[Ax,mode] * alphaG_local_xyz_mat[Ax,Bx] * Lq[Bx,mode]
            alphaG_local_pairwise[A,B] += val

    alphaG_local_pairwise[:] = np.add(alphaG_local_pairwise, alphaG_local_pairwise.T)
    for A in range(Natom):
        alphaG_local_pairwise[A,A] /= 2.0

    wn = wavenum[mode]
    wavenumst = ("%10.3f" % wn) if wn > 0 else (("%9.3f" % -wn) + 'i')

    pr("-----------------------------------------------------\n")
    pr("              ROA Normal Mode Decomposition          \n")
    pr("  Mode: %d       Harmonic Freq.: %10s              \n" % (mode+1, wavenumst))
    pr("-----------------------------------------------------\n")
    pr(" Atom Pair     alpha*G        Beta(G)^2    Beta(A)^2 \n")
    pr("-----------------------------------------------------\n")

    alphaG_local_pairwise *= roa_conv
    beta_G2_local_pairwise *= roa_conv
    beta_A2_local_pairwise *= roa_conv

    total_alphaG = 0.0
    total_betaG2 = 0.0
    total_betaA2 = 0.0

    for A in range(Natom):
        for B in range(A+1):
            pr("(%2d,%2d):%15.5f%15.5f%15.5f\n" % (A,B,
                alphaG_local_pairwise[A,B],
                beta_G2_local_pairwise[A,B],
                beta_A2_local_pairwise[A,B]) )
            total_alphaG += alphaG_local_pairwise[A,B] 
            total_betaG2 += beta_G2_local_pairwise[A,B]
            total_betaA2 += beta_A2_local_pairwise[A,B]

    pr("-----------------------------------------------------\n")
    pr("Totals  %15.5f%15.5f%15.5f\n" % (total_alphaG, total_betaG2, total_betaA2))
    pr("-----------------------------------------------------\n")
    pr("\nTop contributors\n")
    pr("-----------------------------------------------------\n")
    pr(" Atom Pair     alpha*G        Beta(G)^2    Beta(A)^2 \n")
    pr("-----------------------------------------------------\n")

    alphaG_dict = {}
    for A in range(Natom):
        for B in range(A+1):
            alphaG_dict[ (A,B) ] = alphaG_local_pairwise[A,B]

    limit = 0.20 * abs(max(alphaG_local_pairwise.min(), alphaG_local_pairwise.max(), key=abs))
    sorted_keys = sorted(alphaG_dict, key=lambda k: abs(alphaG_dict[k]), reverse=True)
    for key in sorted_keys:
        if abs(alphaG_dict[key]) > limit:
            pr("(%2d,%2d):%15.5f\n" % (
                  key[0], key[1], alphaG_dict[key] ))

    betaG2_dict = {}
    for A in range(Natom):
        for B in range(A+1):
            betaG2_dict[ (A,B) ] = beta_G2_local_pairwise[A,B]

    limit = 0.20 * abs(max(beta_G2_local_pairwise.min(), beta_G2_local_pairwise.max(), key=abs))

    sorted_keys = sorted(betaG2_dict, key=lambda k: abs(betaG2_dict[k]), reverse=True)
    for key in sorted_keys:
        if abs(betaG2_dict[key]) > limit:
            pr("(%2d,%2d):               %15.5f\n" % (
                   key[0], key[1], betaG2_dict[key]))

    betaA2_dict = {}
    for A in range(Natom):
        for B in range(A+1):
            betaA2_dict[ (A,B) ] = beta_A2_local_pairwise[A,B]

    limit = 0.20 * abs(max(beta_A2_local_pairwise.min(), beta_A2_local_pairwise.max(), key=abs))

    sorted_keys = sorted(betaA2_dict, key=lambda k: abs(betaA2_dict[k]), reverse=True)
    for key in sorted_keys:
        if abs(betaA2_dict[key]) > limit:
            pr("(%2d,%2d):                              %15.5f\n" % (
                  key[0], key[1], betaA2_dict[key]))
    pr("-----------------------------------------------------\n")


    return


# Tensor inner product
def zeta(A, B):
    value = 0.0
    for i in range(3):
        for j in range(3):
            value += 0.5 * (3.0 * A[i, j] * B[i, j] - A[i, i] * B[j, j])
    return value

# For definition of robustness parameters phi and psi, see 
# dx.doi.org/10.1021/ct500697e | J. Chem. Theory Comput. 2014, 10, 5520−5527
def f_robustnessPhi(A,G):
    cosPhi = zeta(A,G) / sqrt( zeta(A,A) * zeta(G,G) )
    phi = acos(cosPhi) * 180.0 / pi
    return phi

def frobeniusInnerProduct(A, B):
    return np.trace(A.dot(B.T)) / sqrt(np.trace(A.dot(A.T)) * np.trace(B.dot(B.T)))

def f_robustnessPsi(alpha,A):
    Atilde = np.zeros( (3,3) )
    for i in range(3):
        for j in range(3):
            for k in range(3):
                for l in range(3):
                    Atilde[i,j] += levi(i,k,l) * A[k][l][j]
    cosPsi = frobeniusInnerProduct(alpha, Atilde)
    angle = acos(cosPsi) * 180.0 / pi
    return angle

