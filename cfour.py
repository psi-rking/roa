import subprocess
import numpy as np
import re
from qcelemental import molutil

class CFOUR(object):

  def format_basis(orig):
      return orig.upper().replace('D-AUG','DA').replace('CC-','')

  def __init__(self, xyz, symbols, wfn, keywords, title="title", executable=None):
    self.xyz = xyz
    self.symbols = symbols
    self.keywords = {}
    for k,v in keywords.items():
        self.keywords[k.upper()] = v.upper() if isinstance(v,str) else v

    keys = self.keywords.keys()
    if 'CALCLEVEL' not in keys:
        self.keywords['CALCLEVEL']=wfn.upper()
    if 'MEMORY_SIZE' not in keys:
        self.keywords['MEMORY_SIZE']=10
    if 'MEM_UNIT' not in keys:
        self.keywords['MEM_UNIT']='GB'
    if 'COORDINATES' not in keys: 
        self.keywords['COORD']='CARTESIAN'
    if 'SCF_CONV' not in keys: 
        self.keywords['SCF_CONV']=11
    if 'SCF_DAMP' not in keys: 
        self.keywords['SCF_DAMP']=100
    if 'SCF_EXPSTART' not in keys: 
        self.keywords['SCF_EXPSTART']=100
    if 'SCF_MAXCYC' not in keys: 
        self.keywords['SCF_MAXCYC']=300
    if 'UNITS' not in keys:  # changing CFOUR default!
        self.keywords['UNITS']='BOHR'

    # The following basis sets are non-native to CFOUR.
    bs_special_to_C4 = [
    '6-31+G*', '6-31++G*', 'asp-cc-pvdz', 'd-aug-cc-pvdz', 'd-aug-cc-pvtz',
    'orp', 'sadlej-lpol-ds', 'sadlej-lpol-dl', 'sadlej-lpol-fs', 'sadlej-lpol-fl',
    '6-311++G(3df,3pd)' ]

    # weird names with special formatting simplified in custom GENBAS
    bs_c4_weird = { '6-311++G(3df,3pd)' : 'POPLE-BIG' }

    if self.keywords['BASIS'] in bs_special_to_C4:
        BS = self.keywords['BASIS']
        if BS in bs_c4_weird:
            self.keywords['SPECIAL_BASIS'] = bs_c4_weird[BS]
        else:
            self.keywords['SPECIAL_BASIS'] = CFOUR.format_basis(BS)
        self.keywords['BASIS'] = 'SPECIAL'
    else:
        self.keywords['BASIS'] = CFOUR.format_basis(self.keywords['BASIS'])
        if 'SPECIAL_BASIS' in self.keywords:
            # print('ignoring SPECIAL_BASIS')
            self.keywords.pop('SPECIAL_BASIS')
    
    self.keywords['COORD'] = 'CARTESIAN'
    self.title = title
    if executable == None:
        self.executable = '/Users/rking/bin/run-cfour'
    else:
        self.executable = 'cfour'


  def run(self, calctype=None, scratchName=None, read_E=False, read_Gx=False):
    if scratchName is None:
        scratchName = self.title.replace(' ','')
    if calctype.upper() == 'GRADIENT':
        if 'DERIV_LEVEL' not in self.keywords:
            self.keywords['DERIV_LEVEL'] = 1
    elif calctype.upper() == 'HESSIAN':
        if 'DERIV_LEVEL' not in self.keywords:
            self.keywords['DERIV_LEVEL'] = 2
        if 'VIB' not in self.keywords:
            self.keywords['VIB'] = 'EXACT'
    self.makeZMAT()
    # print("Launching CFOUR executable...")
    # returns CompletedProcess()
    rc = subprocess.run([self.executable, scratchName], capture_output=True)
    if rc.returncode != 0:
        print(rc.stdout)
        print(rc.stderr)
        raise Exception('Bad return code from CFOUR')
    if calctype.upper() == 'ENERGY':
        return self.parseFinalEnergyFromOutput()
    elif calctype.upper() == 'GRADIENT':
        E = self.parseFinalEnergyFromOutput()
        (Zs, coord, grad) = self.parseGRD()
        rmsd, mill = molutil.B787(self.xyz, coord, None, None,
          mols_align=1e-12, atoms_map=True, verbose=False)
        RotMat = mill.rotation
        grad2 = np.dot(mill.rotation, grad.T).T
        diff = np.max(np.abs(grad - grad2))
        grad[:] = grad2
        #print(f'Max change from alignment of gradient {diff:10.5e}')
        #x = np.dot(mill.rotation, coord.T).T # reproduces self.xyz
        return E, grad
    elif calctype.upper() == 'HESSIAN':
        c4h = self.read_hessian()  # read the hessian from FCMFINAL
        #print('original H')
        #print(c4h)
        coord = self.parseGeometryFromOutput()
        rmsd, mill = molutil.B787(coord, self.xyz, None, None,
          mols_align=1e-12, atoms_map=True, verbose=False)
        c4h2 = mill.align_hessian(c4h)
        diff = np.max(np.abs(c4h - c4h2))
        c4h[:] = c4h2
        #print(f'Max change from alignment of Hessian {diff:10.5e}')

        if read_E:
            E = self.parseFinalEnergyFromOutput()
        if read_Gx:
            (Zs, coord, grad) = self.parseGRD()
            rmsd, mill = molutil.B787(self.xyz, coord, None, None,
              mols_align=1e-12, atoms_map=True, verbose=False)
            RotMat = mill.rotation
            grad[:] = np.dot(mill.rotation, grad.T).T
        if read_E and read_Gx:
            return (E, grad, c4h)
        elif read_E:
            return (E, c4h)
        elif read_Gx:
            return (grad, c4h)
        else:
            return c4h
    return rc


  def makeZMAT(self, fp=None):
    s = self.title + '\n'
    for i in range(len(self.symbols)):
        s += "%s%17.12f%17.12f%17.12f\n" % (self.symbols[i],
               self.xyz[i][0],self.xyz[i][1],self.xyz[i][2])
    s += '\n' + "*CFOUR("

    # gets ignored anyway, but lets remove it for tidyness
    zmat_keywords = self.keywords.copy()
    if 'SPECIAL_BASIS' in zmat_keywords:
        zmat_keywords.pop('SPECIAL_BASIS')

    tab = 0
    cnt = 0
    for k,v in zmat_keywords.items():
        cnt += 1
        s += "%s=%s" % (k,v)
        tab += 1
        if cnt == len(zmat_keywords):
            s += ')\n'
        elif tab == 5:
            s += '\n'
            tab = 0
        else:
            s += ','
    s += '\n'

    if self.keywords['BASIS'] == 'SPECIAL':
        for at in self.symbols:
          s += "%s:%s\n" %(at, self.keywords['SPECIAL_BASIS'])
        s += '\n'

    f = open('ZMAT','w')
    f.write(s)
    f.close()
    return

  def read_hessian(self, infile='FCMFINAL.out'):
      Natom = len(self.symbols)
      H = np.zeros( (3*Natom*Natom,3) )
      with open(infile,'r') as f:
          f.readline()
          for row, line in enumerate(f):
            v = line.split()
            H[row][0] = v[0]
            H[row][1] = v[1]
            H[row][2] = v[2]
      H = H.reshape( (3*Natom,3*Natom) )
      return H

  def parseDIPDER(self):
      Natom = len(self.symbols)
      mu_x = np.zeros( (Natom,3) )
      mu_y = np.zeros( (Natom,3) )
      mu_z = np.zeros( (Natom,3) )
      with open('DIPDER','r') as f:
          f.readline()
          for at in range(Natom):
            v = f.readline().split()
            mu_x[at][0] = v[1]
            mu_x[at][1] = v[2]
            mu_x[at][2] = v[3]
          f.readline()
          for at in range(Natom):
            v = f.readline().split()
            mu_y[at][0] = v[1]
            mu_y[at][1] = v[2]
            mu_y[at][2] = v[3]
          f.readline()
          for at in range(Natom):
            v = f.readline().split()
            mu_z[at][0] = v[1]
            mu_z[at][1] = v[2]
            mu_z[at][2] = v[3]
      return (mu_x, mu_y, mu_z)

  def parseGeometryFromOutput(self):
      with open('output.dat','r') as f:
          outtext = f.read()
          mobj = re.search(
              r'^\s+' + r'Symbol    Number           X              Y              Z' + r'\s*' +
              r'^\s+(?:-+)\s*' +
              r'((?:\s+[A-Z]+\s+[0-9]+\s+[-+]?\d+\.\d+\s+[-+]?\d+\.\d+\s+[-+]?\d+\.\d+\s*\n)+)' +
              r'^\s+(?:-+)\s*',
              outtext, re.MULTILINE)
          if mobj:
              xyz = np.zeros( (len(mobj.group(1).splitlines()), 3) )
              for row, line in enumerate(mobj.group(1).splitlines()):
                  lline = line.split()
                  xyz[row][0] = lline[-3]
                  xyz[row][1] = lline[-2]
                  xyz[row][2] = lline[-1]
              return xyz

  def writeFile15(self, H, outfile='file15.dat'):
      Natom = len(self.symbols)
      H2 = H.reshape((3*Natom*Natom,3))
      s = ''
      with open(outfile,'w') as f15:
          for row in range(len(H2)):
              s += "%20.10f%20.10f%20.10f\n" % (H2[row][0],H2[row][1],H2[row][2])
          f15.write(s)
      return

  def writeFile17(self, mu_x, mu_y, mu_z, outfile='file17.dat'):
      Natom = len(self.symbols)
      s = ''
      with open(outfile, 'w') as f17:
          for row in range(len(mu_x)):
              s += "%20.10f%20.10f%20.10f\n" % (mu_x[row][0],mu_x[row][1],mu_x[row][2])
          for row in range(len(mu_y)):
              s += "%20.10f%20.10f%20.10f\n" % (mu_y[row][0],mu_y[row][1],mu_y[row][2])
          for row in range(len(mu_z)):
              s += "%20.10f%20.10f%20.10f\n" % (mu_z[row][0],mu_z[row][1],mu_z[row][2])
          f17.write(s)
      return

  def parseGRD(self):
      Natom = len(self.symbols)
      geom = np.zeros( (Natom,3) )
      grad = np.zeros( (Natom,3) )
      Zs   = np.zeros( (Natom) )

      with open('GRD','r') as f: 
          v = f.readline() 
          for atom in range(Natom):
            v = f.readline().split()
            Zs[atom]      = float(v[0])
            geom[atom][0] = float(v[1])
            geom[atom][1] = float(v[2])
            geom[atom][2] = float(v[3])
          for atom in range(Natom):
            v = f.readline().split()
            grad[atom][0] = float(v[1])
            grad[atom][1] = float(v[2])
            grad[atom][2] = float(v[3])
      return (Zs, geom, grad)
  
  def parseFinalEnergyFromOutput(self, fname='output.dat'):
      rval = 0.0
      with open(fname,'r') as f:
          for line in f:
              v = line.split()
              if len(v) > 4:
                  if v[0:5] == ['The','final','electronic','energy','is']:
                      rval=float(v[5])
      return rval

