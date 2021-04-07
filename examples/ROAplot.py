#Examples of how to use the roa SPECTRUM class
import roa

# Read a dictionary file and print the ROA parameters.
sp = roa.SPECTRUM()
sp.readDictionaryOutputFile('cc2-631Gs.sp.out')
# (For backward compatibility with old psi4), read ROA parameters from 
# an old psi4 output file.
# sp.readPsi4OutputFile('psi-out.dat')
print(sp)

# Plot a Raman plus ROA spectrum
# Include all the modes:
#R = range(0,len(sp.data['Frequency'))

# Include selected modes
R = []
for Inu, nu in enumerate(sp.data['Frequency']):
    if nu > 500:
        R.append(Inu)

peaksRaman = [(sp.data['Frequency'][i], sp.data['Raman Intensity (circular)'][i]) for i in R] 

ROA_types = ['ROA R-L Delta(180)', 'ROA R-L Delta(90)_z', 'ROA R-L Delta(90)_x', 'ROA R-L Delta(0)']
ROA_data = []

for T in ROA_types:
    T_data = []
    for i in R:
        nu = sp.data['Frequency'][i]
        T_data.append( (nu, sp.data[T][i]) )
    ROA_data.append(T_data)

#roa.plot.plotSpectrum([peaksRaman], ["Raman"], peakType='Gaussian',
#plotStyleList=['b.-'], Xmin=1000, Xmax=4000)

roa.plot.plotROAspectrum(peaksRaman, 'Raman Intensity (circular)', ROA_data, ROA_types,
Xmin=200, Xmax=4000, plotStyleList=['b.-','g.-','r.-','k.-','m.-']
,title='6-31G* CC2 HOOH', peakType='Lorentzian', delta=10.0, Npoints=500)

