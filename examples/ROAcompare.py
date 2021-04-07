import roa
import numpy as np

# Originally, I was comparing peak heights but not rigorously matching
# vibrational modes (as others have done in literature)..
# This shows how to compare peaks
# Compares peaks by Average Deviation, RMS Deviation, and Average Relative
# Deviation. Ignores very small peaks.
#roa.compareSpectraFiles('ccpvtz.sp.out', ['631Gs.sp.out', 'ccpvdz.sp.out'])

# Now trying comparison by comparing the discretized (with modelled)
# width spectrum itself.

XminDefault = 500
XmaxDefault = 3500
NptsDefault = 500
peakTypeDefault = 'Lorentzian'
widthDefault = 15

def computeDiffArea(peaks1, peaks2, Xmin=XminDefault, Xmax=XmaxDefault, Npts=NptsDefault,
    width=widthDefault, peakType=peakTypeDefault):
    """ Function to compute area of absolute difference between two spectra. """
    x = np.linspace(Xmin, Xmax, Npts)
    y1 = roa.discretizedSpectrum(x, peaks1, width, peakType)
    y2 = roa.discretizedSpectrum(x, peaks2, width, peakType)
    DX = (Xmax-Xmin) / Npts
    diff = abs(y2-y1)
    area = DX * np.sum(diff)
    return area

def computeAbsArea(peaks, Xmin=XminDefault, Xmax=XmaxDefault, Npts=NptsDefault,
    width=widthDefault, peakType=peakTypeDefault):
    """ Function to compute area of absolute difference between two spectra. """
    x = np.linspace(Xmin, Xmax, Npts)
    y = roa.discretizedSpectrum(x, peaks, width, peakType)
    DX = (Xmax-Xmin) / Npts
    diff = abs(y)
    area = DX * np.sum(diff)
    return area

fileRef = 'cc2-ccpvdz.out'

print(f'Reference File: {fileRef:s}')
spRef = roa.SPECTRUM()
spRef.readPsi4OutputFile(fileRef)

filesToTest = [
  'cc2-sto3g.out',
  'cc2-ccpvdz.out',
  'cc2-631Gs.out'
 ]

print('Testing files:')
spTest = {}
for f in filesToTest:
    print(f'\t{f:s}')
    s = roa.SPECTRUM()
    s.readPsi4OutputFile(f)
    spTest[f] = s

#Select independent variable (probably frequency), and properties to test.
variableX = 'Frequency'
#spectraToTest = ['IR Intensity', 'Raman Intensity (circular)', 'ROA R-L Delta(180)',
#          'ROA R-L Delta(90)_z', 'ROA R-L Delta(90)_x', 'ROA R-L Delta(0)']
spectraToTest = ['IR Intensity']

areaDiff = {}
for spectrumType in spectraToTest:
    peaks1 = list(zip(spRef.data[variableX], spRef.data[spectrumType]))
    areaDiff[spectrumType] = {}
    areaDiff[spectrumType + ' Relative'] = {}

    for f in filesToTest:
        peaks2 = list(zip(spTest[f].data[variableX], spTest[f].data[spectrumType]))
        areaDiff[spectrumType][f] = computeDiffArea(peaks1, peaks2)
        #print('file: %s' % f )
        #print(areaDiff[spectrumType][f])
        areaDiff[spectrumType + ' Relative'][f] = areaDiff[spectrumType][f] / computeAbsArea(peaks1)

for spectrumType in spectraToTest:
    print(f'Area of Absolute Difference for {spectrumType:s}:')
    sortedProperty = sorted(areaDiff[spectrumType].items(), key=lambda x: x[1], reverse=True)
    for entries in sortedProperty:
        print(f'{entries[0]:20s}{entries[1]:10.5f}')

for spectrumType in spectraToTest:
    print(f'Area of Percent Difference for {spectrumType:s}:')
    sortedProperty = sorted(areaDiff[spectrumType + ' Relative'].items(),
                            key=lambda x: x[1], reverse=True)
    for entries in sortedProperty:
        print(f'{entries[0]:20s}{100*entries[1]:>10.2f}')

