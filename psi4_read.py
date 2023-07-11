"""
    Functions to read 
      1) Cartesian Hessian from file 15.
      2) Dipole derivatives from file 17.
      3) From displacement directories, (e.g. 3-y-p) read
          a) frequency dependent polarizability.
          b) electric dipole/magnetic dipole polarizability
             (i.e., the optical rotation tensor).
          c) electric dipole/quadrupole polarizability.

    in the names of variables there is a lack of consistency.  These
    are equivalent rot = G;pol = alpha
"""
import numpy as np
import os

def psi4_read_hessian(Natom):
    H = np.zeros( (3*Natom*Natom, 3) )
    with open("file15.dat", "r") as fHessian:
        for i in range(3*Natom*Natom):
            s = fHessian.readline().split()
            for xyz in range(3):
                H[i][xyz] = float(s[xyz])
    H.shape = (3*Natom, 3*Natom)
    # not quite symmetric; we could symmetrize
    return H

def check_symmetric(a, rtol=1e-05, atol=1e-08):
    return np.allclose(a, a.T, rtol=rtol, atol=atol)

# Read in the Dipole-Moment Derivatives from file 17.
def psi4_read_dipole_derivatives(Natom):
    dipder = np.zeros( (3, 3*Natom) )
    with open("file17.dat", "r") as fMu:
        for i in range(3):  # d(mu_x), d(mu_y), d(mu_z)
            for j in range(Natom):
                s = fMu.readline().split()
                for xyz in range(3):
                    dipder[i][3*j+xyz] = float(s[xyz])
    return dipder

# Read in fd_pol, the displaced (1) electric-dipole/electric-dipole
# polarizability from subdirectories.
def psi4_read_polarizabilities(dirNames, omega):
    root_dir = os.getcwd()
    line_cnt = 0
    reading = False
    found = False
    fd_pol = []
    for d in dirNames:
                os.chdir(root_dir + '/' + d)
                one_pol = np.zeros( (3,3) )
                with open('output.dat', "r") as fpol:
                    for line in fpol:
                        if reading: line_cnt += 1
                        words = line.split()
                        if len(words) > 4:
                            if words[1] == 'Dipole' and words[2] == 'Polarizability':
                                reading = True
                                # total hack because CCSD output has a line of
                                # dashes lacking for CC2
                                if words[0] == 'CCSD':
                                    line_cnt = -1
                        if line_cnt == 1:
                            if (abs(omega)-float(words[4]) > 1e-5):
                                reading = False
                                line_cnt = 0
                        elif line_cnt in [6,7,8]:
                            for a in range(3):
                                one_pol[line_cnt-6][a] = float(words[1+a])
                        elif line_cnt == 9:
                            line_cnt = 0
                            reading = False
                            found = True

                if not found:
                    raise Exception("Can't find Dipole Polarizability for %s:%f" %(d,omega))
                fd_pol.append(one_pol)
                line_cnt = 0
                reading = False
    os.chdir(root_dir)
    return fd_pol

# Read in fd_rot, the displaced (2) electric-dipole/magnetic-dipole
# polarizability from subdirectories.
def psi4_read_optical_rotation_tensor(dirNames, omega, gauge='Modified Velocity'):
    root_dir = os.getcwd()
    reading = False
    found = False
    line_cnt = 0
    fd_rot = []
    for d in dirNames:
                os.chdir(root_dir + '/' + d)
                one_rot = np.zeros( (3,3) )
                with open('output.dat', "r") as frot:
                    for line in frot:
                        words = line.split()
                        if reading: line_cnt += 1
                        if len(words) > 5:
                            if words[1] == 'Optical' and words[2] == 'Rotation' and \
                            words[3] == 'Tensor':
                                loc_gauge = words.index('Gauge):')
                                out_gauge = " ".join(words[4:loc_gauge])
                                out_gauge = "".join(c for c in out_gauge if c not in '():')
                                if gauge == out_gauge:
                                    reading = True
                                    line_cnt = 0
                        if line_cnt == 2:
                            if (abs(omega)-float(words[4]) > 1e-5):
                                reading = False
                                line_cnt = 0
                                Found = False
                        elif line_cnt in [7,8,9]:
                            for a in range(3):
                                one_rot[line_cnt-7][a] = float(words[1+a])
                        elif line_cnt == 10:
                            line_cnt = 0
                            reading = False
                            found = True
                if not found:
                    raise Exception("Can't find Optical Rotation Tensor for %s:%f" %(s,omega))
                fd_rot.append(one_rot)
    os.chdir(root_dir)
    return fd_rot

# Read in fd_quad, the displaced (3) electric-dipole/electric-quadrupole
# polarizability from subdirectories.
def psi4_read_dipole_quadrupole_polarizability(dirNames, omega):
    root_dir = os.getcwd()
    reading = False
    found = False
    line_cnt = 0
    fd_quad = []
    for d in dirNames:
                os.chdir(root_dir + '/' + d)
                one_quad = np.zeros( (9,3) )
                with open('output.dat', "r") as fquad:
                    for line in fquad:
                        words = line.split()
                        if reading: line_cnt += 1
                        if len(words) > 4:
                            if words[1] == 'Electric-Dipole/Quadrupole' and words[2] == 'Polarizability':
                                reading = True
                                line_cnt = 0
                        if line_cnt == 2:
                            if (abs(omega)-float(words[4]) > 1e-5):
                                reading = False
                                line_cnt = 0
                        elif line_cnt in [7,8,9]:
                            for a in range(3):
                                one_quad[a][line_cnt-7] = float(words[1+a])
                        elif line_cnt in [13,14,15]:
                            for a in range(3):
                                one_quad[a+3][line_cnt-13] = float(words[1+a])
                        elif line_cnt in [19,20,21]:
                            for a in range(3):
                                one_quad[a+6][line_cnt-19] = float(words[1+a])
                        elif line_cnt == 22:
                            line_cnt = 0
                            reading = False
                            found = True
                if not found:
                    raise Exception("Can't find Dipole-Quadrupole Polarizability for %s:%f" %(s,omega))
                fd_quad.append(one_quad)
    os.chdir(root_dir)
    return fd_quad

# This is weird, but often times I am writing a ZMAT to include a geometry
# from psi4, so this is a hack to get the canonical psi4 geom.
def psi4_read_ZMAT(Natom):
    geom = np.zeros( (Natom,3) )
    with open("ZMAT", "r") as fZMAT:
        fZMAT.readline()
        for i in range(Natom):
            s = fZMAT.readline().split()
            for xyz in range(3):
                geom[i][xyz] = float(s[1+xyz])
    return geom

