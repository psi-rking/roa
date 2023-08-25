import psi4
import optking
import roa
import os

do_optimize    = False
do_hessian     = False
do_roa_tensors = True
do_roa_tensors_only_missing = False
do_roa_analysis = True

mol = psi4.geometry("""
  0 1
  O        1.3848525400        0.1153987700       -0.0542277600
  O       -1.3848525400       -0.1153987700       -0.0542277600
  H        1.7747609300       -1.4290967500        0.8606338400
  H       -1.7747609300        1.4290967500        0.8606338400
  units au
""")

MYLABEL='psi4-hooh'
CCwfn='cc2'

BASIS_SET_OPT='cc-pvdz'
BASIS_SET_HESS='cc-pvdz'
BASIS_SET_CCTENSOR='sto-3g'

FROZEN_CORE_OPT=False
FROZEN_CORE_HESS=False
FROZEN_CORE_CCTENSOR=True # psi4 needs True/False or 1/0

OMEGA=[532,'nm']
VIB_MODES = [3,4]  # 'cart', 'all', or [1, 3, ...] for 1st/3rd/.. highest vu,
GAUGE='length'

R_CONVERGENCE=1e-8     # convergence of CC amplitudes
CCTENSOR_DISP_SIZE=0.05
CCTENSOR_R_CONVERGENCE=1e-5  # EP's data supports this choice
ANALYZE_MODES_PAIRWISE=[1] # Modes to analyze pairwise; '1' is highest freq
CCTENSOR_MEM='3 GB'
CCTENSOR_NSIMULTANEOUS=4

psi4.set_memory('12 GB')
calcLbl=BASIS_SET_CCTENSOR + '-' + CCwfn + '-' + MYLABEL
psi4.core.set_output_file(calcLbl + '.out')

myROA = roa.Psi4ROA.ROA(mol, pr=psi4.core.print_out)

# Optimize geometry
if do_optimize:
    psi4_options = {
        'basis' : BASIS_SET_OPT,
        'freeze_core' : FROZEN_CORE_OPT,
        'r_convergence': R_CONVERGENCE,
    }
    psi4.set_options(psi4_options)
    optGeom, json_opt_output = myROA.optimize(CCwfn, prog='psi4')
    if os.path.exists('opt_log.out'):
        os.rename('opt_log.out', calcLbl + '-' + 'opt.log')
    psi4.core.clean()
else:
    myROA.analysis_geom = mol.geometry().to_array()

# Compute Hessian
if do_hessian:
    psi4_options = {
      'basis' : BASIS_SET_HESS,
      'freeze_core' : FROZEN_CORE_HESS,
      'r_convergence': R_CONVERGENCE,
    }
    psi4.set_options(psi4_options)
    myROA.compute_hessian(CCwfn, 'psi4')
    myROA.compute_dipole_derivatives(CCwfn, 'psi4')
    psi4.core.clean()

# Compute tensors
psi4_options = {
  'r_convergence': CCTENSOR_R_CONVERGENCE,
  'basis' : BASIS_SET_CCTENSOR,
  'omega' : OMEGA,
  'gauge' : GAUGE,
  'freeze_core': FROZEN_CORE_CCTENSOR
}
psi4.set_options(psi4_options)

# Initialize database for finite-differences of tensors.
myROA.fd_db_init(CCwfn, 'roa', ['roa_tensor'], OMEGA, CCTENSOR_DISP_SIZE)

myROA.make_coordinate_vectors(VIB_MODES)

psi4.set_memory(CCTENSOR_MEM)

myROA.fd_db_make_displaced_geoms()
myROA.fd_db_show()

if do_roa_tensors:
    if do_roa_tensors_only_missing:
        myROA.fd_db_generate_inputs(CCwfn, overwrite=False)
    else:
        myROA.fd_db_generate_inputs(CCwfn, overwrite=True)

    myROA.fd_db_run(psi4.executable, nThreads=CCTENSOR_NSIMULTANEOUS)

if do_roa_analysis:
    # Currently, analyze_ROA reads from files
    # file15.dat nuclear second derivatives
    # file17.dat dipole moment derivatives
    # For displaced geometries, e.g., 1_x_p/output.dat,
    # a) electric-dipole electric-dipole polarizability;
    # b) electric-dipole electric-quadrupole polarizability;
    # c) electric dipole magnetic dipole polarizability (optical rotation tensor)
    # 
    myROA.analyze_ROA(calcLbl, GAUGE, analyzeModesPairwise=ANALYZE_MODES_PAIRWISE)

myROA.close_fd_db()

