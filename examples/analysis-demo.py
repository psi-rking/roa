# Compare area between two spectra.
import roa
import numpy as np
import matplotlib.pyplot as plt
import os

baseDir  = os.getcwd() + '/'
refFile  = 'aug-cc-pVQZ-cc2-hooh-vroa.sp.out'
testFile = 'aug-cc-pVDZ-cc2-hooh-vroa.sp.out'

spRef = roa.SPECTRUM.fromDictionaryOutputFile(baseDir + refFile)
spTest = roa.SPECTRUM.fromDictionaryOutputFile(baseDir + testFile)

title = "HOOH, CC2/aug-cc-pVQZ hessian"
labels = ['QZ', 'DZ']

# *** plotSpectra
#def plotSpectra(spectra, spectrumType=None, labels=None, Npts=1000,
#    Xmin=400, Xmax=4000, plotStyleList=None, title="Spectrum", width=10,
#    peakType='Lorentzian', fill_between=False):

# Show default behavior
roa.plotSpectra([spRef, spTest])

# blue circle solid, # green diamond dotted
roa.plotSpectra([spRef, spTest], 'ROA R-L Delta(180)', ['Ref','Test'],
 1000, 1000, 2000, ['bo-','gD:'], 'Ref vs Test', 10, 'Lorentzian', False)

# *** Fill between with 2 plots
roa.plotSpectra([spRef, spTest], 'ROA R-L Delta(180)', ['Ref','Test'],
 1000, 1000, 2000, ['bo-','gD:'], 'Ref vs Test', 10, 'Lorentzian', True)

# *** plotPeaks
# def plotPeaks(peakList, labels=None, Npts=200, Xmin=None, Xmax=None,
#     plotStyleList=None, title="Spectrum", width=10, peakType='Lorentzian'):

# Can prune peaks if desired
ROAtype = 'ROA R-L Delta(180)'
peaks1 = zip(spRef.data['Frequency'], spRef.data[ROAtype])
peaks2 = zip(spTest.data['Frequency'], spTest.data[ROAtype])

# red X dash-dot, green point solid
roa.plotPeaks([peaks1,peaks2], labels=['Ref','Test'], Npts=1000,
Xmin=1000, Xmax=2000, plotStyleList=['rx-.','g.-'], title="plotPeaks demo",
width=10, peakType='Lorentzian')

# *** plotROAspectrum
# def plotROAspectrum(peaksRaman, labelRaman, peakList, labels=None,
#   Npts=1000, Xmin=None, Xmax=None, title='', width=10, peakType='Lorentzian',
#   plotStyleList=None):

peaksRaman = zip(spRef.data['Frequency'], spRef.data['Raman Intensity (linear)'])
ROAtypes=['ROA R-L Delta(180)','ROA R-L Delta(90)_z']
peakList = []
for r in ROAtypes:
    peakList.append(zip(spTest.data['Frequency'], spTest.data[r]))

roa.plotROAspectrum(peaksRaman, 'Raman Intensity (linear)', peakList,
labels=ROAtypes, Npts=1000, Xmin=400, Xmax=4000, title='ROA spectrum',
width=10, peakType='Lorentzian',plotStyleList=None)

# *** compareSpectra
#def compareBroadenedSpectra(spRef, spTest, spectrumType=None, Npts=2000,
#    Xmin=400, Xmax=4000, width=10, peakType='Lorentzian', comparisonType='RADF',
#    symmetrize=False, printLevel=0):

#default behavior, Delta(180), RADF, returns value
radf = roa.compareBroadenedSpectra(spRef, spTest)
print(f"RADF: {radf:10.4f}")

roa.compareBroadenedSpectra(spRef, spTest, comparisonType='RADF', printLevel=1)
roa.compareBroadenedSpectra(spRef, spTest, comparisonType='SNO', printLevel=1)
roa.compareBroadenedSpectra(spRef, spTest, comparisonType='DNO', printLevel=1)
roa.compareBroadenedSpectra(spRef, spTest, comparisonType='IDF', printLevel=1)

print('* Limited range')
roa.compareBroadenedSpectra(spRef, spTest, comparisonType='IDF', Xmin=1000,
    Xmax=2000, width=10, peakType='Lorentzian', printLevel=1)

print('* Test symmetrize')
rt = roa.compareBroadenedSpectra(spRef, spTest, comparisonType='SNO')
print(f"<Ref |SNO|Test>: {rt:10.4f}")
tr = roa.compareBroadenedSpectra(spTest, spRef, comparisonType='SNO')
print(f"<Test|SNO| Ref>: {tr:10.4f}")
print(f"Average        : {0.5*(tr+rt):10.4f}")
tval = roa.compareBroadenedSpectra(spTest, spRef, comparisonType='SNO', symmetrize=True)
print(f"Average (auto) : {tval:10.4f}")

# * test invert
print('\n* Test inversion, compare spectrum with its inverted version')
spRef_inverted = roa.SPECTRUM.fromDictionaryOutputFile(baseDir + refFile) 
spRef_inverted.invert()

for comp in ['RADF','SNO','DNO','IDF']:
    tval = roa.compareBroadenedSpectra(spRef, spRef_inverted, comparisonType=comp)
    print("<sp|" + comp + "|i(sp)>:" + f"{tval:10.4f}")

