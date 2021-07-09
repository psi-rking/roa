import psi4
import optking
import roa
import os

# Basis sets not in CFOUR canonical set, must be marked 'SPECIAL' in input.
bs_special_to_C4 = [
'6-31+G*', '6-31++G*', 'asp-cc-pvdz', 'd-aug-cc-pvdz', 'd-aug-cc-pvtz',
'orp', 'sadlej-lpol-ds', 'sadlej-lpol-dl', 'sadlej-lpol-fs', 'sadlej-lpol-fl',
'6-311++G(3df,3pd)' ]

# unpredictable c4 basis-set renames
bs_c4_weird = { '6-311++G(3df,3pd)' : 'POPLE-BIG' }

# Basis set specific keywords for C4.  With lots of diffuse functions doesn't
# converge so lowering.
bs_C4_keywords = {
  'sadlej-lpol-fl' : {'SCF_CONV':7} # CFOUR SCF convergence stinks
  # 'MEM_UNIT':'GB', 'MEMORY_SIZE': 200,  # default is round(core.get_memory()//1e9)
}
bs_psi4_keywords = {
  'sadlej-lpol-fl' : {'r_convergence' : 5e-7} # loosen ccresponse conv
}

basis_sets = [
'3-21G', '6-31+G*', '6-31++G*', 'asp-cc-pvdz', 'aug-cc-pvdz', 'cc-pvdz',
'cc-pvtz', 'd-aug-cc-pvdz', 'orp', 'aug-cc-pvtz', 'd-aug-cc-pvtz', 'sadlej-lpol-ds',
'sadlej-lpol-dl', 'sadlej-lpol-fs', 'sadlej-lpol-fl', '6-311++G(3df,3pd)' ]

for bs in basis_sets:
    print(f"Starting {bs} basis set.")

    mol = psi4.geometry("""
    0 1
    O   0.7266046846   0.0573586059  -0.0310546152
    O  -0.7266046846  -0.0573586059  -0.0310546152
    H   0.9428086264  -0.7195465859   0.4990531021
    H  -0.9428086264   0.7195465859   0.4990531021
    """)

    psi4.core.set_output_file(bs+'-cc2.out')
    psi4.set_memory('200 GB')

    omega = [532, 'nm']
    gauge = 'VELOCITY'
    psi4_options = {
      'basis' : bs,
      'omega' : omega,
      'gauge' : gauge
    }
    if bs in bs_psi4_keywords.keys():
        psi4_options.update(bs_psi4_keywords[bs])
    psi4.set_options(psi4_options)

    # Initialize ROA manager object.
    myROA = roa.Psi4ROA.ROA(mol, pr=psi4.core.print_out)

    # Set geometry.
    optGeom, json_opt_output = myROA.optimize('cc2')
    # or, if not optimizing:
    # myROA.analysis_geom = np.array([ insert xyz coordinates ])

    # basis getting cleared out?
    psi4.set_options(psi4_options)

    # Compute Hessian, generate file15
    extraC4keywords = {}

    if bs in bs_special_to_C4: # format of ZMAT differs
        if bs in bs_c4_weird:
            extraC4keywords = {'BASIS':'SPECIAL','SPECIAL_BASIS':bs_c4_weird[bs]}
        else:
            extraC4keywords = {'BASIS':'SPECIAL','SPECIAL_BASIS':bs}

    if bs in bs_C4_keywords.keys():
        extraC4keywords.update(bs_C4_keywords[bs])

    myROA.compute_hessian('cc2','cfour', c4executable='/home/rking/bin/run-cfour-21',
        c4kw = extraC4keywords)

    # or,
    # myROA.compute_hessian('cc2','psi4')

    psi4.set_memory('40 GB')

    # Compute dipole moment derivatives, generate file17
    # Actually, this reads (not computes) from DIPDER
    myROA.compute_dipole_derivatives('cc2','cfour')
    # or,
    #myROA.compute_dipole_derivatives('cc2','psi4')

    # Initialize database for finite-differences of tensors.
    myROA.fd_db_init('cc2', 'roa', ['roa_tensor'], omega, 0.005)

    #vib_modes = [1,2,3] # To do only the 3 normal modes of highest frequency
    vib_modes = [i+1 for i in range(6)]
    myROA.make_coordinate_vectors(vib_modes)
    # or,
    # to do all 3N Cartesian displacements
    #myROA.make_coordinate_vectors()

    # Generate the displaced geometries and the input files
    myROA.fd_db_make_displaced_geoms()
    myROA.fd_db_show() #see what's in there, if desired
    myROA.fd_db_generate_inputs('cc2', overwrite=True)

    # Run the fd computations
    myROA.fd_db_run(psi4.executable, nThreads=6)

    # Currently, analyze_ROA reads from files
    # file15.dat nuclear second derivatives
    # file17.dat dipole moment derivatives
    # For displaced geometries, e.g., 1_x_p/output.dat,
    # a) electric-dipole electric-dipole polarizability;
    # b) electric-dipole electric-quadrupole polarizability;
    # c) electric dipole magnetic dipole polarizability (optical rotation tensor)
    myROA.analyze_ROA('cc2test', gauge)
    myROA.close_fd_db()

    os.rename('output.dat', bs+'-cfour-cc2.out')
    os.remove('fd-database.dat')
    os.remove('fd-database.dir')
    os.remove('file15.dat')
    os.remove('DIPDER')
    os.remove('file17.dat')

