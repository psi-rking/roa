#Examples of how to use the roa SPECTRUM class
import roa

# Read a dictionary file and print the ROA parameters.
#dunning = roa.SPECTRUM()
#dunning.readDictionaryOutputFile('ccpvdz.sp.out')
#print(dunning)

# (For backward compatibility with old psi4), read ROA parameters from 
# an old psi4 output file.
#pople = roa.SPECTRUM()
#pople.readPsi4OutputFile('psi-out.dat')
#print(pople)

# Compare the peak heights in a list of spectra to those in a reference spectrum.
# Compares peaks by Average Deviation, RMS Deviation, and Average Relative Deviation
# Ignores very small peaks.
#roa.compareSpectraFiles('ccpvtz.sp.out', ['631Gs.sp.out', 'ccpvdz.sp.out'])

# Plot a Raman plus ROA spectrum with 3 modes for HOOH
#sp = roa.SPECTRUM()
#sp.readDictionaryOutputFile('ccpvdz.sp.out')
#R = range(0,3)
#peaksRaman = [(sp.data['Frequency'][i], sp.data['Raman Intensity (circular)'][i]) for i in R] 
#peaks180   = [(sp.data['Frequency'][i], sp.data['ROA R-L Delta(180)'][i]) for i in R]
#Z90    = [(sp.data['Frequency'][i], sp.data['ROA R-L Delta(90)_z'][i]) for i in R]
#X90    = [(sp.data['Frequency'][i], sp.data['ROA R-L Delta(90)_x'][i]) for i in R]
#peaks0 = [(sp.data['Frequency'][i], sp.data['ROA R-L Delta(0)'][i]) for i in R]

#roa.plot.plotSpectrum([peaksRaman], ["Raman"], peakType='Gaussian',
#plotStyleList=['b.-'], Xmin=1000, Xmax=4000)

#roa.plot.plotROAspectrum(peaksRaman, 'Raman Intensity', [peaks180, Z90, X90, peaks0],
#labels=['R-L Delta(180)', 'R-L Delta(90)z', 'R-L Delta(90)x', 'R-L Delta(0)'],
#Xmin=1200, Xmax=4000, plotStyleList=['b.-','g.-','r.-','k.-','m.-']
#,title='ccpvdz CC2 hooh', peakType='Lorentzian', delta=10.0, Npoints=500)

