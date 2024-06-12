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

def computeArea(peaks, Xmin, Xmax, Npts, width, peakType):
    """ Function to compute area. """
    x = np.linspace(Xmin, Xmax, Npts)
    y = roa.discretizedSpectrum(x, peaks, width, peakType)
    DX = (Xmax-Xmin) / Npts
    area = DX * np.sum(y)
    return area

def computeAbsArea(peaks, Xmin, Xmax, Npts, width, peakType):
    """ Function to compute area of absolute difference between two spectra. """
    x = np.linspace(Xmin, Xmax, Npts)
    y = roa.discretizedSpectrum(x, peaks, width, peakType)
    DX = (Xmax-Xmin) / Npts
    diff = abs(y)
    area = DX * np.sum(diff)
    return area

def computeSquaredArea(peaks, Xmin, Xmax, Npts, width, peakType):
    """ Function to compute area of square of function """
    x = np.linspace(Xmin, Xmax, Npts)
    y = roa.discretizedSpectrum(x, peaks, width, peakType)
    DX = (Xmax-Xmin) / Npts
    area = DX * np.sum(y**2)
    return area

def computeDiffArea(peaks1, peaks2, Xmin, Xmax, Npts, width, peakType):
    """ Function to compute area of difference between two spectra. """
    x = np.linspace(Xmin, Xmax, Npts)
    y1 = roa.discretizedSpectrum(x, peaks1, width, peakType)
    y2 = roa.discretizedSpectrum(x, peaks2, width, peakType)
    DX = (Xmax-Xmin) / Npts
    area = DX * np.sum(y2-y1)
    return area

def computeProductArea(peaks1, peaks2, Xmin, Xmax, Npts, width, peakType):
    """ Function to compute area of difference between two spectra. """
    x = np.linspace(Xmin, Xmax, Npts)
    y1 = roa.discretizedSpectrum(x, peaks1, width, peakType)
    y2 = roa.discretizedSpectrum(x, peaks2, width, peakType)
    DX = (Xmax-Xmin) / Npts
    area = DX * np.sum(y1*y2)
    return area

def computeAbsDiffArea(peaks1, peaks2, Xmin, Xmax, Npts, width, peakType):
    """ Function to compute area of absolute difference between two spectra. """
    x = np.linspace(Xmin, Xmax, Npts)
    y1 = roa.discretizedSpectrum(x, peaks1, width, peakType)
    y2 = roa.discretizedSpectrum(x, peaks2, width, peakType)
    DX = (Xmax-Xmin) / Npts
    area = DX * np.sum(abs(y2-y1))
    return area

# comparisonType:
# SNO, single-normalized overlap I(fs fr) / I(fr^2)                  -inf, +inf
# DNO, doubly-normalized overlap I(fs fr) / sqrt[ I(fs^2)*I(fr^2)]     -1, +1
# IDF, integrated difference function [I(fr^2) - I(fs^2)] / I(fr^2)  -inf, +1
# RADF,relative absolute difference function I[ |fs - fc| ] / I(fr)  -inf, +inf

def compareBroadenedSpectra(fileRef, filesToTest, spectraToTest, Xmin=100,
    Xmax=4000, Npts=1000, peakType='Lorentzian', width=15, comparisonType='RADF'):

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
        areaDiff[spectrumType] = {}

        if comparisonType == 'RADF':
            peaksRef = list(zip(spRef.data[variableX], spRef.data[spectrumType]))
            areaAbsRef = computeAbsArea(peaksRef, Xmin, Xmax, Npts, width, peakType)
            areaDiff[spectrumType]['Abs Area of Ref'] = areaAbsRef
            print(f'Absolute Area of reference spectrum {areaAbsRef:10.5e}')
            areaDiff[spectrumType + ' Relative'] = {}

            for f in filesToTest:
                peaks2 = list(zip(spTest[f].data[variableX], spTest[f].data[spectrumType]))
                areaDiff[spectrumType][f] = computeAbsDiffArea(peaksRef, peaks2, Xmin, Xmax, Npts, width, peakType)
                areaDiff[spectrumType + ' Relative'][f] = areaDiff[spectrumType][f] / areaAbsRef

            #for spectrumType in spectraToTest:
            #    print(f'Area of Absolute Difference for {spectrumType:s}:')
            #    sortedProperty = sorted(areaDiff[spectrumType].items(), key=lambda x: x[1], reverse=True)
            #    for entries in sortedProperty:
            #        print(f'{entries[0]:>30s}{entries[1]:>10.5f}')

        elif comparisonType == 'SNO':
            peaksRef = list(zip(spRef.data[variableX], spRef.data[spectrumType]))
            areaSqRef = computeSquaredArea(peaksRef, Xmin, Xmax, Npts, width, peakType)
            areaDiff[spectrumType]['Area(Square(Ref))'] = areaSqRef
            print(f'Area of Ref^2 {areaSqRef:10.5e}')
            areaDiff[spectrumType + ' Relative'] = {}

            for f in filesToTest:
                peaks2 = list(zip(spTest[f].data[variableX], spTest[f].data[spectrumType]))
                areaDiff[spectrumType][f] = computeProductArea(peaksRef, peaks2, Xmin, Xmax, Npts, width, peakType)
                areaDiff[spectrumType + ' Relative'][f] = areaDiff[spectrumType][f] / areaSqRef

        elif comparisonType == 'DNO':
            peaksRef = list(zip(spRef.data[variableX], spRef.data[spectrumType]))
            areaSqRef = computeSquaredArea(peaksRef, Xmin, Xmax, Npts, width, peakType)
            areaDiff[spectrumType]['Area(Square(Ref))'] = areaSqRef
            print(f'Area of Ref^2 {areaSqRef:10.5e}')
            areaDiff[spectrumType + ' Relative'] = {}

            for f in filesToTest:
                peaks2 = list(zip(spTest[f].data[variableX], spTest[f].data[spectrumType]))
                areaSqTest = computeSquaredArea(peaks2, Xmin, Xmax, Npts, width, peakType)
                areaDiff[spectrumType][f] = computeProductArea(peaksRef, peaks2, Xmin, Xmax, Npts, width, peakType)
                areaDiff[spectrumType + ' Relative'][f] = areaDiff[spectrumType][f] / np.sqrt(areaSqTest*areaSqRef)

        elif comparisonType == 'IDF':
            peaksRef = list(zip(spRef.data[variableX], spRef.data[spectrumType]))
            areaSqRef = computeSquaredArea(peaksRef, Xmin, Xmax, Npts, width, peakType)
            areaDiff[spectrumType]['Area(Square(Ref))'] = areaSqRef
            print(f'Area of Ref^2 {areaSqRef:10.5e}')
            areaDiff[spectrumType + ' Relative'] = {}

            for f in filesToTest:
                peaks2 = list(zip(spTest[f].data[variableX], spTest[f].data[spectrumType]))
                areaSqTest = computeSquaredArea(peaks2, Xmin, Xmax, Npts, width, peakType)
                areaDiff[spectrumType + ' Relative'][f] = (areaSqRef-areaSqTest)/areaSqRef

    
    for spectrumType in spectraToTest:
        print(f'Comparison via {comparisonType} for {spectrumType:s}:')
        sortedProperty = sorted(areaDiff[spectrumType + ' Relative'].items(),
                                key=lambda x: x[1], reverse=True)
        for entries in sortedProperty:
            print(f'{entries[0]:>30s}{entries[1]:>10.4f}')
    return areaDiff


