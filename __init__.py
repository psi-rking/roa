from .psi4_read import psi4_read_hessian
from .psi4_read import psi4_read_dipole_derivatives
from .psi4_read import psi4_read_polarizabilities
from .psi4_read import psi4_read_optical_rotation_tensor
from .psi4_read import psi4_read_dipole_quadrupole_polarizability
from .psi4_read import psi4_read_ZMAT
from .scatter import scatter
from .scatter import omega_in_au
from .spectrum import SPECTRUM
from .compare import compareOutputFiles, compareSpectraFiles, reorder
from .mode_scatter import modeVectors, modeScatter
from sys import platform
from .plot import discretizedSpectrum
from . import Psi4ROA
try:
  import matplotlib
  if platform != 'darwin':
    matplotlib.use('TkAgg')
  from .plot import plotSpectrum, plotROAspectrum
except ImportError:
  pass

