import subprocess
import numpy as np
import re

class CFOUR(object):

  def format_basis(orig):
      return orig.upper().replace('D-AUG','DA').replace('CC-','')

  def __init__(self, xyz, symbols, keywords, title, executable=None):
    self.xyz = xyz
    self.symbols = symbols
    self.keywords = keywords.copy()

    if self.keywords['BASIS'] == 'SPECIAL':
        if 'SPECIAL_BASIS' not in self.keywords:
            raise Exception('Must provide SPECIAL_BASIS')
        self.keywords['SPECIAL_BASIS'] = CFOUR.format_basis(self.keywords['SPECIAL_BASIS'])
    else:
        self.keywords['BASIS'] = CFOUR.format_basis(self.keywords['BASIS'])
        if 'SPECIAL_BASIS' in self.keywords:
            # print('ignoring SPECIAL_BASIS')
            self.keywords.pop('SPECIAL_BASIS')
    
    self.keywords['COORD'] = 'CARTESIAN'
    self.title = title
    self.executable = 'cfour' if executable is None else executable

  def run(self, scratchName=None):
    if scratchName is None:
        scratchName = self.title.replace(' ','')
    self.makeZMAT()
    # print("Launching CFOUR executable...")
    # returns CompletedProcess()
    rc = subprocess.run([self.executable, scratchName], capture_output=True)
    if rc.returncode != 0:
        print(rc.stdout)
        print(rc.stderr)
        raise Exception('Bad return code from CFOUR')

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

  def parseHessian(self):
      Natom = len(self.symbols)
      H = np.zeros( (3*Natom*Natom,3) )
      with open('FCMFINAL.out','r') as f:
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

  def parseGeometry(self):
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

  def writeFile15(self, H):
      Natom = len(self.symbols)
      H2 = H.reshape((3*Natom*Natom,3))
      s = ''
      with open('file15.dat','w') as f15:
          for row in range(len(H2)):
              s += "%20.10f%20.10f%20.10f\n" % (H2[row][0],H2[row][1],H2[row][2])
          f15.write(s)
      return

  def writeFile17(self, mu_x, mu_y, mu_z):
      Natom = len(self.symbols)
      s = ''
      with open('file17.dat','w') as f17:
          for row in range(len(mu_x)):
              s += "%20.10f%20.10f%20.10f\n" % (mu_x[row][0],mu_x[row][1],mu_x[row][2])
          for row in range(len(mu_y)):
              s += "%20.10f%20.10f%20.10f\n" % (mu_y[row][0],mu_y[row][1],mu_y[row][2])
          for row in range(len(mu_z)):
              s += "%20.10f%20.10f%20.10f\n" % (mu_z[row][0],mu_z[row][1],mu_z[row][2])
          f17.write(s)
      return


