import psi4
import optking
import roa

mol = psi4.geometry("""
  0 1
  O         1.3165700332        0.0940585320       -0.0206039146
  O        -1.3165700332       -0.0940585320       -0.0206039146
  H         1.7968722880       -1.3634001708        0.9079549572
  H        -1.7968722880        1.3634001708        0.9079549572
  units bohr
 """)

psi4.core.set_output_file('psi-out.dat')
psi4.set_memory('4 GB')

omega = [532, 'nm']
gauge = 'VELOCITY'
psi4_options = {
  'basis' : 'cc-pvdz',
  'omega' : omega,
  'gauge' : gauge,
}
psi4.set_options(psi4_options)

# Initialize ROA manager object.
myROA = roa.Psi4ROA.ROA(mol, pr=psi4.core.print_out)

# Set geometry.
myROA.optimize('cc2') # returns geometry, if desired
# or, if not optimizing
# myROA.analysis_geom = optgeom

# Compute Hessian, generate file15
myROA.compute_hessian('cc2','cfour')
# or,
# myROA.compute_hessian('cc2','psi4')

# Compute dipole moment derivatives, generate file17
# Actually, this reads (not computes) from DIPDER
myROA.compute_dipole_derivatives('cc2','cfour')
# or,
#myROA.compute_dipole_derivatives('cc2','psi4')

# Initialize database for finite-differences of tensors.
myROA.fd_db_init('cc2', 'roa', ['roa_tensor'], omega, 0.005)

vib_modes = [1,2,3] # To do only the 3 normal modes of highest frequency
myROA.make_coordinate_vectors(vib_modes)
# or,
# to do all 3N Cartesian displacements
#myROA.make_coordinate_vectors()

# Generate the displaced geometries and the input files
myROA.fd_db_make_displaced_geoms()
myROA.fd_db_show() #see what's in there, if desired
myROA.fd_db_generate_inputs('cc2', overwrite=False)

# Run the fd computations
myROA.fd_db_run(psi4.executable, nThreads=5)

# Currently, analyze_ROA reads from files
# file15.dat nuclear second derivatives
# file17.dat dipole moment derivatives
# For displaced geometries, e.g., 1_x_p/output.dat,
# a) electric-dipole electric-dipole polarizability;
# b) electric-dipole electric-quadrupole polarizability;
# c) electric dipole magnetic dipole polarizability (optical rotation tensor)
myROA.analyze_ROA('cc2test', gauge)
myROA.close_fd_db()

