import psi4
import optking
import roa
import os

do_optimize    = True
do_hessian     = True
do_roa_tensors = True
do_roa_tensors_only_missing = False
do_roa_analysis = True

mol = psi4.geometry("""
  0 1
  O        1.4000        0.1153987700       -0.0542277600
  O       -1.4000       -0.1153987700       -0.0542277600
  H        1.7747609300       -1.4290967500        0.8606338400
  H       -1.7747609300        1.4290967500        0.8606338400
  units au
  no_reorient
  no_com
""")

MYLABEL='cfour-hooh'
CCwfn='cc2'

BASIS_SET_OPT="cc-pvdz"
BASIS_SET_HESS="cc-pvdz"
BASIS_SET_CCTENSOR="cc-pvdz"

FROZEN_CORE_OPT='ON'
FROZEN_CORE_HESS='ON'
FROZEN_CORE_CCTENSOR=True # psi4 needs True/False or 1/0

OMEGA=[532,'nm']
VIB_MODES = 'all'  # 'cart', 'all', or [1, 3, ...] for 1st/3rd/.. highest vu,
GAUGE='length'

CCTENSOR_DISP_SIZE=0.05
CCTENSOR_R_CONVERGENCE=1e-5  # EP's data supports this choice
ANALYZE_MODES_PAIRWISE=[1,2] # Modes to analyze pairwise; '1' is highest freq
CCTENSOR_MEM='3 GB'
CCTENSOR_NSIMULTANEOUS=4

psi4.set_memory('12 GB')
calcLbl=BASIS_SET_CCTENSOR + '-' + CCwfn + '-' + MYLABEL
psi4.core.set_output_file(calcLbl + '.out')

myROA = roa.Psi4ROA.ROA(mol, pr=psi4.core.print_out)

if do_optimize:
    c4_options = {
      'CALC' : CCwfn,
      'BASIS' : BASIS_SET_OPT,
      'FROZEN_CORE' : FROZEN_CORE_OPT
    }
    optking_options = {
      'g_convergence' : 'gau_verytight'
    }
    optGeom, json_opt_output = myROA.optimize(CCwfn, 'cfour', c4_options, optking_options)
    if os.path.exists('opt_log.out'):
        os.rename('opt_log.out', 'opt.log')
    if os.path.exists('output.dat'):
        os.rename('output.dat', 'opt-output.dat')
    c4_options.clear()
else:
    myROA.analysis_geom = mol.geometry().to_array()

# Compute Hessian, generate file15
if do_hessian:
    c4_options = {
      'CALC' : CCwfn,
      'BASIS' : BASIS_SET_HESS,
      'FROZEN_CORE' : FROZEN_CORE_HESS
    }
    myROA.compute_hessian(CCwfn, 'cfour', c4_options)
    myROA.compute_dipole_derivatives(CCwfn, 'cfour', c4kw=c4_options) # reads file17
    c4_options.clear()

# Setup and run finite-difference optical tensors.
psi4_options = {
  'r_convergence': CCTENSOR_R_CONVERGENCE,
  'basis' : BASIS_SET_CCTENSOR,
  'omega' : OMEGA,
  'gauge' : GAUGE,
  'freeze_core': FROZEN_CORE_CCTENSOR
}
psi4.set_options(psi4_options)

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
    myROA.analyze_ROA(calcLbl, GAUGE, analyzeModesPairwise=ANALYZE_MODES_PAIRWISE)

myROA.close_fd_db()
