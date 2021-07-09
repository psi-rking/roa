import roa
import numpy as np

# ----------
# Compare spectra by numerically broadening them, and comparing
# the resulting curves.

def readROAfromFile(filename, psi4_dict=True):
    print(f'Reading File: {filename:s}')
    sp = roa.SPECTRUM()
    if psi4_dict == True:
        sp.readDictionaryOutputFile(filename)
    else:
        sp.readPsi4OutputFile(filename)
    return sp

def computeDiffArea(peaks1, peaks2, Xmin, Xmax, Npts, width, peakType):
    """ Function to compute area of absolute difference between two spectra. """
    x = np.linspace(Xmin, Xmax, Npts)
    y1 = roa.discretizedSpectrum(x, peaks1, width, peakType)
    y2 = roa.discretizedSpectrum(x, peaks2, width, peakType)
    DX = (Xmax-Xmin) / Npts
    diff = abs(y2-y1)
    area = DX * np.sum(diff)
    return area

def computeAbsArea(peaks, Xmin, Xmax, Npts, width, peakType):
    """ Function to compute area of absolute difference between two spectra. """
    x = np.linspace(Xmin, Xmax, Npts)
    y = roa.discretizedSpectrum(x, peaks, width, peakType)
    DX = (Xmax-Xmin) / Npts
    print(f'Xmin:{Xmin:10.2f}, Xmax:{Xmax:10.2f}, Npts:{Npts:10d}, DX:{DX:10.5f}')
    diff = abs(y)
    area = DX * np.sum(diff)
    return area

def compareBroadenedSpectra(fileRef, filesToTest, spectraToTest, Xmin=100,
    Xmax=4000, Npts=1000, peakType='Lorentzian', width=15):

    print('Loading Reference Spectrum:')
    spRef = readROAfromFile(fileRef, fileRef[-6:] == 'sp.out')
    print(spRef)

    print('Loading Test Spectra:')
    spTest = {}
    for f in filesToTest:
        spTest[f] = readROAfromFile(f, f[-6:] == 'sp.out')

    #Select independent variable (probably frequency), and properties to test.
    variableX = 'Frequency'

    areaDiff = {}
    for spectrumType in spectraToTest:
        peaks1 = list(zip(spRef.data[variableX], spRef.data[spectrumType]))
        #print(peaks1)
        areaDiff[spectrumType] = {}
        areaDiff[spectrumType + ' Relative'] = {}
        areaAbsRef = computeAbsArea(peaks1, Xmin, Xmax, Npts, width, peakType)
        print(f'Absolute Area of reference spectrum {areaAbsRef:10.5f}')

        for f in filesToTest:
            peaks2 = list(zip(spTest[f].data[variableX], spTest[f].data[spectrumType]))
            areaDiff[spectrumType][f] = computeDiffArea(peaks1, peaks2, Xmin, Xmax, Npts, width, peakType)
            areaDiff[spectrumType + ' Relative'][f] = areaDiff[spectrumType][f] / areaAbsRef

    for spectrumType in spectraToTest:
        print(f'Area of Absolute Difference for {spectrumType:s}:')
        sortedProperty = sorted(areaDiff[spectrumType].items(), key=lambda x: x[1], reverse=True)
        for entries in sortedProperty:
            print(f'{entries[0]:>30s}{entries[1]:>10.5f}')
    
    for spectrumType in spectraToTest:
        print(f'Area of Percent Difference for {spectrumType:s}:')
        sortedProperty = sorted(areaDiff[spectrumType + ' Relative'].items(),
                                key=lambda x: x[1], reverse=True)
        for entries in sortedProperty:
            print(f'{entries[0]:>30s}{100*entries[1]:>10.2f}')
    return


