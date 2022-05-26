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
pc_au2amu          = constants.au2amu
pc_dipmom_debye2si = constants.dipmom_debye2si
pc_amu2kg          = constants.amu2kg
pc_na              = constants.na
pc_c               = constants.c
pc_e0              = constants.e0
pc_bohr2angstroms  = constants.bohr2angstroms
pc_hartree2J       = constants.hartree2J
pc_bohr2m          = constants.bohr2m
pc_me              = constants.me
pc_h               = constants.h
pc_hartree2ev      = constants.hartree2ev
km_convert         = pc_hartree2J / (pc_bohr2m * pc_bohr2m * pc_amu2kg * pc_au2amu)
cm_convert         = 1.0 / (2.0 * pi * pc_c * 100.0)
from psi4.core import print_out, BasisSet
from psi4.driver import qcdb

def modeVectorsQCDB(mol, hessian, modesToReturn=None, print_lvl=1, pr=print):
    Natom = mol.natom()
    geom = mol.geometry().to_array()
    masses = np.asarray([mol.mass(at) for at in range(Natom)])
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

    # pr(vibtext)
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
# Stolen from qcdb; make col phase canonical
def _phase_cols_to_max_element(arr, tol=1.e-2, verbose=1):
    ""Returns copy of 2D `arr` scaled such that, within cols, max(fabs)
    element is positive. If max(fabs) is pos/neg pair, scales so first
    element (within `tol`) is positive.
    ""
    arr2 = np.copy(arr)

    rephasing = []
    for v in range(arr.shape[1]):
        vextreme = 0.0
        iextreme = None

        # find most extreme value
        for varr in arr[:, v]:
            vextreme = max(np.absolute(varr), vextreme)

        # find the first index whose fabs equals that value, w/i tolerance
        for iarr, varr in enumerate(arr[:, v]):
            if (vextreme - np.absolute(varr)) < tol:
                iextreme = iarr
                break

        sign = np.sign(arr[iextreme, v])
        if sign == -1.:
            rephasing.append(str(v))
        arr2[:, v] *= sign

    if rephasing and verbose >= 2:
        print('Negative modes rephased:', ', '.join(rephasing))

    return arr2


# Return the vectors along which to displace for finite differences
# and ROA analysis.
def modeVectors(geom, masses, hessian, modesToReturn=None, print_lvl=1, pr=print):
        # geom,      # (Natom,3) Cartesian coordinates of nuclei
        # masses,    # atomic masses
        # hessian    # (3*Natom,3*Natom) Cartesian 2nd derivative
    Natom = len(geom)

    pr("\n\tCalculating normal modes; Input coordinates (bohr)\n")
    for a in range(Natom):
        pr("\t%15.10f%15.10f%15.10f\n" % (geom[a,0],geom[a,1],geom[a,2]))

    # Now construct the rotational and translational coordinates to project
    # out of the Cartesian Hessian.
    comGeom = centerGeometry(geom, masses)
    mwGeom = np.zeros( (Natom,3) )
    for i in range(Natom):
        for xyz in range(3):
            mwGeom[i][xyz] = comGeom[i][xyz] * sqrt(masses[i])

    pr("\n\tAtomic Masses for Raman Computation:\n")
    for i in range(Natom):
        pr("\t%5d %12.8f\n" % (i+1, masses[i]) )

    if print_lvl > 1:
        pr("\n\tMass-Weighted Coordinates relative to COM\n")
        for a in range(Natom):
            pr("\t%15.10f%15.10f%15.10f\n" % (mwGeom[a,0],mwGeom[a,1],mwGeom[a,2]))

    Iinv = inertiaTensorInverse(comGeom, masses)
    if print_lvl > 1:
        pr("\n\tInertia Tensor Inverse\n")
        pr(str(Iinv)+'\n')

    # Generate 6 rotation and translation vectors to project out of Hessian
    P = np.zeros( (3*Natom, 3*Natom) )
    total_mass = np.sum(masses)

    for i in range(3*Natom):
        icart = i % 3
        iatom = i // 3
        imass = masses[iatom]
        P[i][i] = 1.0

        for j in range(3*Natom):
            jcart = j % 3
            jatom = j // 3
            jmass = masses[jatom]
            P[i,j] +=  -1.0 * sqrt(imass * jmass) / total_mass * (icart == jcart)

            for a in range(3):
                for b in range(3):
                    for c in range(3):
                        for d in range(3):
                            P[i,j] +=  -1.0 * levi(a, b, icart) * mwGeom[iatom,b] * Iinv[a,c] * \
                                              levi(c, d, jcart) * mwGeom[jatom,d]

    # Mass-weight the Hessian.  Will be [Eh/(bohr^2 au)]
    M = np.zeros( (3*Natom, 3*Natom) )
    for i in range(Natom):
        for j in range(3):
            M[3*i + j, 3*i + j] = 1.0 / sqrt((masses[i]) / pc_au2amu)

    T = np.dot(M, hessian)
    F = np.dot(T, M)

    # Project out rotational and translational degrees of freedom from M-W Hessian.
    T[:] = np.dot(F, P)
    F[:] = np.dot(P.T, T)

    if print_lvl > 2:
        pr("\n\tMass-weighted, projected Hessian\n")
        pr(str(F)+'\n')

    Fevals, Fevecs = symmMatEig(F) # rows of Fevecs are eigenvectors

    # Resort for descending evals
    idx = np.argsort(Fevals)[::-1] # descending evals
    Fevals[:] = Fevals[idx]
    Fevecs[:] = Fevecs[idx, :]
    # use function that phase normalizes the columns
    Fevecs = (_phase_cols_to_max_element(Fevecs.T, tol=1.e-2, verbose=1)).T
    # matches qcdb: vibinfo['q'] == Datum('normal mode', 'a0 u^1/2',
    #  qL, comment='normalized mass-weighted')

    # Unmass-weighting destroys normalization going to au^-1/2.
    Lx = np.dot(M, Fevecs.T)
    # matches qcdb: vibinfo['w'] = 
    # Datum('normal mode', 'a0', wL, comment='un-mass-weighted')
    # apart from mass unit

    if print_lvl > 2:
        pr("\n\tNormal transform matrix u^(-1/2) * Hmw_evects\n")
        pr(str(Lx)+'\n')

    freqs    = np.zeros(3*Natom)
    for i in range(3*Natom):
        if Fevals[i] < 0.0:
            freqs[i] = -1.0 * cm_convert * sqrt(abs(km_convert * Fevals[i]))
        else:
            freqs[i] = cm_convert * sqrt(km_convert * Fevals[i])

    pr("\n\t----------------------\n")
    pr("\t Mode #   Harmonic    \n")
    pr("\t            Freq.     \n")
    pr("\t           (cm-1)     \n")
    pr("\t----------------------\n")
    for i in range(0,3*Natom):
        if freqs[i] < 0.0:
            pr("\t  %3d   %10.2fi\n" % (i, abs(freqs[i])))
        else:
            pr("\t  %3d   %10.2f \n" % (i, freqs[i]))

    selected_modes = np.zeros( (len(modesToReturn),3*Natom))
    selected_freqs = np.zeros( len(modesToReturn) )

    for i, mode in enumerate(modesToReturn):
        selected_modes[i,:] = Lx[:,mode]
        selected_freqs[i] = freqs[mode]

    pr("\nFrequencies (with modes) returned from modeVectors.\n")
    pr(str(selected_freqs)+'\n')

    if print_lvl > 1:
        pr("\nVectors returned from modeVectors are rows.\n")
        pr(str(selected_modes)+'\n')

    return (selected_modes, selected_freqs)

def symmMatEig(mat):
    try:
        evals, evects = np.linalg.eigh(mat)
        # apply convention for consistency
        if abs(min(evects[:,0])) > abs(max(evects[:,0])):
            evects[:,0] *= -1.0
    except:
        raise Exception("symmMatEig: could not compute eigenvectors")
    evects[:] = evects.T  # return eigenvectors as rows
    return evals, evects

def inertiaTensorInverse(geom, masses):
    I = np.zeros( (3,3) )
    for i in range(len(masses)):
        I[0,0] += masses[i]*(geom[i,1]*geom[i,1]+geom[i,2]*geom[i,2])
        I[1,1] += masses[i]*(geom[i,0]*geom[i,0]+geom[i,2]*geom[i,2])
        I[2,2] += masses[i]*(geom[i,0]*geom[i,0]+geom[i,1]*geom[i,1])
        I[0,1] -= masses[i]*geom[i,0]*geom[i,1]
        I[0,2] -= masses[i]*geom[i,0]*geom[i,2]
        I[1,2] -= masses[i]*geom[i,1]*geom[i,2]

    I[1,0] = I[0,1]
    I[2,0] = I[0,2]
    I[2,1] = I[1,2]
    Iinv = np.linalg.inv(I)
    return Iinv

def centerGeometry(geom, masses):
    total = np.sum(masses)

    com = np.zeros( (3) )
    for i in range(len(masses)):
        for xyz in range(3):
           com[xyz] += masses[i] * geom[i,xyz]
    com[:] = com / total
    centeredGeom = np.zeros( geom.shape )
    for i in range(len(masses)):
        centeredGeom[i,:] = np.subtract( geom[i,:], com )
    return centeredGeom
"""

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
        pr=print
      ):

    Natom = len(geom)
    Nmodes = len(mode_indices)
    pr("\nInput coordinates (bohr)\n")
    for a in range(Natom):
        pr("%15.10f%15.10f%15.10f\n" % (geom[a,0],geom[a,1],geom[a,2]))

    # COMPUTE TENSOR DERIVATIVES
    # A_grad = Array of length # of Cartesian coordinates with respect
    # to which derivative is taken.  Each member is a 3x3 ndarray for the
    #  various electric-dipole/electric-dipole polarizability components.
    A_grad = []
    for i in range(0, len(A_fd), 2):
        grad_mat = np.subtract( A_fd[i], A_fd[i+1] )
        grad_mat[:] /=  (2.0 * step)
        A_grad.append(grad_mat)

    # G_grad = Array of length # of Cartesian coordinates with respect
    # to which derivative is taken.  Each member is a 3x3 ndarray for the
    #  various electric-dipole/magnetic-dipole polarizability components.
    G_grad = []
    for i in range(0, len(G_fd), 2):
        grad_mat = np.subtract( G_fd[i], G_fd[i+1] )
        grad_mat[:] /=  (2.0 * step)
        G_grad.append(grad_mat)

    # Q_grad = Array of length # of Cartesian coordinates with respect
    # to which derivative is taken.  Each member is a 9x3 ndarray for the
    #  various electric quadropole/electric-dipole polarizability components.
    Q_grad = []
    for i in range(0, len(G_fd), 2):
        grad_mat = np.subtract( Q_fd[i], Q_fd[i+1] )
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
        #with open("tender.dat", "w") as f:
        #    f.write(s)

    #pr("\n\tAtomic Masses for Raman Computation:\n")
    #for i in range(Natom):
    #    pr("\t%5d %12.8f\n" % (i+1, masses[i]) )
    # total_mass = np.sum(masses)

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
    mu_der_conv = pc_dipmom_debye2si * pc_dipmom_debye2si / (1e-20 * pc_amu2kg * pc_au2amu) \
                  * pc_na * pi / (3.0 * pc_c * pc_c * 4.0 * pi * pc_e0 * 1000.0)

    IRint = np.zeros(Nmodes)
    for i in range(Nmodes):
        for j in range(3):
            IRint[i] += mu_der_conv * mu_der_q[j,i] * mu_der_q[j,i]

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

    # For each vibrational mode, compute lots of Raman scattering activities
    raman_conv = pc_bohr2angstroms * pc_bohr2angstroms * pc_bohr2angstroms * \
                 pc_bohr2angstroms / pc_au2amu

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
                            abs(freqs[i]), IRint[i], redmass[i], alpha[i] * alpha[i] * raman_conv,
                            beta_alpha2[i] * raman_conv, ramint_linear[i] * raman_conv, depol_linear[i],
                            ramint_circular[i] * raman_conv, depol_circular[i]))
        else:
            pr( "%3d %9.2f %9.4f    %7.4f  %9.4f  %9.4f  %9.4f  %9.4f  %9.4f  %9.4f\n" % (i+1,
                            freqs[i], IRint[i], redmass[i], alpha[i] * alpha[i] * raman_conv,
                            beta_alpha2[i] * raman_conv, ramint_linear[i] * raman_conv, depol_linear[i],
                            ramint_circular[i] * raman_conv, depol_circular[i]) )
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

    cvel = pc_c * pc_me * pc_bohr2angstroms * 1e-10 / (pc_h / (2.0 * pi))
    roa_conv = raman_conv * 1e6 / cvel;
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
        delta_0[i]   = raman_conv * 1e3 * 4.0 * (
          180.0 * alpha[i] * G[i] + 4.0 * beta_G2[i] - 4.0 * betaA2[i]) / cvel
        delta_180[i] = raman_conv * 1e3 * 4.0 * (
          24.0 * beta_G2[i] + 8.0 * betaA2[i]) / cvel
        delta_x[i]   = raman_conv * 1e3 * 4.0 * (
          45.0 * alpha[i] * G[i] + 7.0 * beta_G2[i] + betaA2[i]) / cvel
        delta_z[i]   = raman_conv * 1e3 * 4.0 * (
           6.0 * beta_G2[i] - 2.0 * betaA2[i]) / cvel

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
    Dout['Raman Intensity (linear)']   = raman_conv * ramint_linear[0:Nmodes]
    Dout['Raman Intensity (circular)'] = raman_conv * ramint_circular[0:Nmodes]
    Dout['ROA R-L Delta(0)']           = delta_0[0:Nmodes]
    Dout['ROA R-L Delta(180)']         = delta_180[0:Nmodes]
    Dout['ROA R-L Delta(90)_x']        = delta_x[0:Nmodes]
    Dout['ROA R-L Delta(90)_z']        = delta_z[0:Nmodes]
    Dout['ROA alpha*G']                = alpha[0:Nmodes] * G[0:Nmodes]
    Dout['ROA Beta(G)^2']              = beta_G2[0:Nmodes]
    Dout['ROA Beta(A)^2']              = betaA2[0:Nmodes]
    with open(ROAdictName, "w") as f:
        f.write(str(Dout))

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

def symmMatEig(mat):
    try:
        evals, evects = np.linalg.eigh(mat)
        # apply convention for consistency
        if abs(min(evects[:,0])) > abs(max(evects[:,0])):
            evects[:,0] *= -1.0
    except:
        raise Exception("symmMatEig: could not compute eigenvectors")
    evects[:] = evects.T  # return eigenvectors as rows
    return evals, evects

def inertiaTensorInverse(geom, masses):
    I = np.zeros( (3,3) )
    for i in range(len(masses)):
        I[0,0] += masses[i]*(geom[i,1]*geom[i,1]+geom[i,2]*geom[i,2])
        I[1,1] += masses[i]*(geom[i,0]*geom[i,0]+geom[i,2]*geom[i,2])
        I[2,2] += masses[i]*(geom[i,0]*geom[i,0]+geom[i,1]*geom[i,1])
        I[0,1] -= masses[i]*geom[i,0]*geom[i,1]
        I[0,2] -= masses[i]*geom[i,0]*geom[i,2]
        I[1,2] -= masses[i]*geom[i,1]*geom[i,2]

    I[1,0] = I[0,1]
    I[2,0] = I[0,2]
    I[2,1] = I[1,2]
    Iinv = np.linalg.inv(I)
    return Iinv

def centerGeometry(geom, masses):
    total = np.sum(masses)

    com = np.zeros( (3) )
    for i in range(len(masses)):
        for xyz in range(3):
           com[xyz] += masses[i] * geom[i,xyz]
    com[:] = com / total
    centeredGeom = np.zeros( geom.shape )
    for i in range(len(masses)):
        centeredGeom[i,:] = np.subtract( geom[i,:], com )
    return centeredGeom


# A_der is (3*Natom,9)
# Q_der is (3*Natom,27)
# evals are vibrational eigenvalues for printing
def local_pairs(A_der, G_der, Q_der, Lx, omega, roa_conv, mode, Feval, pr):
    Natom = len(Lx)//3
    L = Lx[:,mode].T

    beta_A2_local_xyz_mat = np.zeros( (3*Natom, 3*Natom) )
    for Ax in range(3*Natom):
        pol_der_A_square = A_der[Ax,:].reshape(3,3)

        for Bx in range(3*Natom):
            quad_der_B_cube = Q_der[Bx,:].reshape(3,3,3)

            value = 0.0
            for i in range(3):
                for j in range(3):
                    for k in range(3):
                        for l in range(3):
                            value += 0.5 * omega * pol_der_A_square[i,j] * levi(i,k,l) * quad_der_B_cube[k,l,j]

            beta_A2_local_xyz_mat[Ax][Bx] = value

    #total = np.dot(L, np.dot(beta_A2_local_xyz_mat, L.T) )
    #pr("\tTotal over A-B (xyz)   : %12.7f\n" % (total * roa_conv))

    beta_A2_local_pairwise = np.zeros( (Natom,Natom) )
    for A in range(Natom):
        for B in range(Natom):
            val = 0
            for Ax in range(3*A, 3*A+3):
                for Bx in range(3*B, 3*B+3):
                    val += L[Ax] * beta_A2_local_xyz_mat[Ax,Bx] * L[Bx]
    
            beta_A2_local_pairwise[A,B] += val

    beta_A2_local_pairwise[:] = np.add(beta_A2_local_pairwise, beta_A2_local_pairwise.T)
    for A in range(Natom):
        beta_A2_local_pairwise[A,A] /= 2.0


    # Compute beta(G')^2 = 1/2[3 * alpha_ij*G'_ij - alpha_ii*G'_jj
    beta_G2_local_xyz_mat = np.zeros( (3*Natom, 3*Natom) )
    for Ax in range(3*Natom):
        pol_der_A_square = A_der[Ax,:].reshape(3,3)
        for Bx in range(3*Natom):
            opt_der_B_square = G_der[Bx,:].reshape(3,3)

            value = 0.0
            for i in range(3):
                for j in range(3):
                    value += 0.5 * (3.0 * pol_der_A_square[i, j] * opt_der_B_square[i, j] - 
                                          pol_der_A_square[i, i] * opt_der_B_square[j, j])
            beta_G2_local_xyz_mat[Ax][Bx] = value

    #total = np.dot(L, np.dot(beta_G2_local_xyz_mat, L.T) )
    #pr("\tTotal over A-B (xyz)   : %12.7f\n" % (total * roa_conv))

    beta_G2_local_pairwise = np.zeros( (Natom,Natom) )
    for A in range(Natom):
        for B in range(Natom):
            val = 0
            for Ax in range(3*A, 3*A+3):
                for Bx in range(3*B, 3*B+3):
                    val += L[Ax] * beta_G2_local_xyz_mat[Ax,Bx] * L[Bx]
   
            beta_G2_local_pairwise[A,B] += val

    beta_G2_local_pairwise[:] = np.add(beta_G2_local_pairwise, beta_G2_local_pairwise.T)
    for A in range(Natom):
        beta_G2_local_pairwise[A,A] /= 2.0

    # Compute alpha*G' = alpha_i * G'_i (can multiply by 45 for forward comparison)
    alphaG_local_xyz_mat = np.zeros( (3*Natom, 3*Natom) )
    for Ax in range(3*Natom):
        pol_der_A_square = A_der[Ax,:].reshape(3,3)
        for Bx in range(3*Natom):
            opt_der_B_square = G_der[Bx,:].reshape(3,3)

            value = 0.0
            for i in range(3):
                for j in range(3):
                    value += pol_der_A_square[i, i] * opt_der_B_square[j, j] / 9.0

            alphaG_local_xyz_mat[Ax][Bx] = value

    #total = np.dot(L, np.dot(alphaG_local_xyz_mat, L.T) )
    #pr("\tTotal       : %12.7f\n" % (total * roa_conv))

    alphaG_local_pairwise = np.zeros( (Natom,Natom) )
    for A in range(Natom):
        for B in range(Natom):
            val = 0
            for Ax in range(3*A, 3*A+3):
                for Bx in range(3*B, 3*B+3):
                    val += L[Ax] * alphaG_local_xyz_mat[Ax,Bx] * L[Bx]

            alphaG_local_pairwise[A,B] += val

    alphaG_local_pairwise[:] = np.add(alphaG_local_pairwise, alphaG_local_pairwise.T)
    for A in range(Natom):
        alphaG_local_pairwise[A,A] /= 2.0

    pr("-----------------------------------------------------\n")
    pr("              ROA Normal Mode Decomposition          \n")
    pr("  Mode: %d       Harmonic Freq.: %9.3f               \n" % (
          mode+1, cm_convert * sqrt(km_convert * Feval)))
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

