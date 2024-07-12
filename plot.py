import numpy as np
import matplotlib.pyplot as plt
from math import pi,sqrt,exp

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

def find_nearest_idx(array, value):
    array = np.asarray(array)
    return (np.abs(array - value)).argmin()

# Gaussian for adding width
# x:     cm^-1 at which to compute
# xzero: center of line
# R:     strength/intensity/area of line
# delta: half-width at 1/e*(peak height); also root(2)*(std. deviation)
# HWHM is root[ln(2)] * delta
def Gaussian(x, xzero, R, Delta, RisPeakHeight=False):
    #xzero = find_nearest(x, xzero)  # make sure our finite representation captures peak
    y = np.zeros(len(x))
    if RisPeakHeight:
        y[:] = [R * exp(-((t-xzero)/Delta)**2) for t in x]
    else:
        y[:] = [R / (Delta * sqrt(pi)) * exp(-((t-xzero)/Delta)**2) for t in x]
    return y

# Lorentzian for adding width
# x:     cm^-1 at which to compute
# xzero: center of line
# R:     strength/intensity/area of line
# gamma: HWHM, half-width at 1/2*(peak height)
def Lorentzian(x, xzero, R, Gamma, RisPeakHeight=False):
    #xzero = find_nearest(x, xzero)  # make sure our finite representation captures peak
    y = np.zeros(len(x))
    if RisPeakHeight:
        y[:] = [R * Gamma**2 / ((t-xzero)**2 + Gamma**2) for t in x]
    else:
        y[:] = [R / pi * Gamma / ((t-xzero)**2 + Gamma**2) for t in x]
    return y

def DeltaPeak(x, xzero, R, Delta=None):
    y = np.zeros(len(x))
    y[find_nearest_idx(x, xzero)] = R
    return y

# x = pts at which evaluation should be done
def discretizedSpectrum(x, peaks, width=10, peakType='Lorentzian'):
    Xmin = min(x)
    Xmax = max(x)
    y = np.zeros(len(x))
    for p in peaks:
        if p[0] > Xmin and p[0] < Xmax:
            y += eval(peakType)(x, p[0], p[1], width)
    return y


def plotSpectra(spectra, spectrumType=None, labels=None, Npts=1000,
    Xmin=400, Xmax=4000, plotStyleList=None, title="Spectrum", width=10,
    peakType='Lorentzian', fill_between=False):

    if fill_between and (len(spectra) != 2):
        raise('fill_between options requires exactly 2 spectra')

    if not spectrumType:
        spectrumType = 'ROA R-L Delta(180)'
    if not labels:
        labels = []
        for s in spectra:
            lbl = s.title
            if lbl[-7:] == '.sp.out':
                lbl = lbl[0:-7]
            labels.append(lbl[lbl.rfind('/')+1:])

    fig = plt.figure()

    ax=fig.add_axes([0.15,0.15,0.70,0.70])
    ax.set_title(title)
    ax.set_xlabel('wavenumber')
    ax.set_ylabel(spectrumType + ' intensity')

    if fill_between: 
        peaks1 = []
        for mode in range(len(spectra[0].data['Frequency'])):
            f = spectra[0].data['Frequency'][mode]
            if f > Xmin and f < Xmax:
                peaks1.append( (f, spectra[0].data[spectrumType][mode]) )
        peaks2 = []
        for mode in range(len(spectra[1].data['Frequency'])):
            f = spectra[1].data['Frequency'][mode]
            if f > Xmin and f < Xmax:
                peaks2.append( (f, spectra[1].data[spectrumType][mode]) )
        x = np.linspace(Xmin, Xmax, Npts)
        y = discretizedSpectrum(x, peaks1, width, peakType)
        y2 = discretizedSpectrum(x, peaks2, width, peakType)

        if plotStyleList is None:
            ax.plot(x, y,'+-')
            ax.plot(x, y2,'+-')
        else:
            ax.plot(x, y, plotStyleList[0])
            ax.plot(x, y2, plotStyleList[1])
        ax.fill_between(x, y, y2)
    else:
        for i, s in enumerate(spectra):
            peaks = []
            for mode in range(len(s.data['Frequency'])):
               f = s.data['Frequency'][mode]
               if f > Xmin and f < Xmax:
                   peaks.append( (f, s.data[spectrumType][mode]) )

            x = np.linspace(Xmin, Xmax, Npts)
            y = discretizedSpectrum(x, peaks, width, peakType)

            if plotStyleList is None:
                ax.plot(x, y,'+-')
            else:
                ax.plot(x, y, plotStyleList[i])

    if labels is not None:
        ax.legend(labels, loc='best')

    plt.xlim(Xmin,Xmax)
    plt.show()
    return


# peakList: tuples of (frequency, intensity)
# width: linewidth parameter (perhaps in cm^-1)
# peakType: may be Lorentzian or Gaussian or DeltaPeak; may be list
def plotPeaks(peakList, labels=None, Npts=1000, Xmin=None, Xmax=None,
plotStyleList=None, title="Spectrum", width=10, peakType='Lorentzian'):

    if type(peakType) is list:
        peakTypes = peakType
    else:
        peakTypes = [peakType for i in range(len(peakList))]

    if Xmin is None:
        Xmin = 1e6
        for l in peakList:
            tmp = max(min(l)[0] - 100,0)
            if tmp < Xmin: Xmin = tmp

    if Xmax is None:
        Xmax = 0
        for l in peakList:
            tmp = max(l)[0] + 100
            if tmp > Xmax: Xmax = tmp

    fig = plt.figure()
    # example of saving to pdf file : plt.savefig('foo.pdf')

    # figure parameters:
    # Figsize (width,height) tuple in inches
    # Dpi         Dots per inches
    # Facecolor   Figure patch facecolor
    # Edgecolor   Figure patch edge color
    # Linewidth   Edge line width
    
    # A plot to fill this figure
    # left, bottom, width, height
    ax=fig.add_axes([0.15,0.15,0.70,0.70])

    ax.set_title(title)
    ax.set_xlabel('wavenumber')
    ax.set_ylabel('intensity')

    for i, l in enumerate(peakList):
        x = np.linspace(Xmin, Xmax, Npts)
        y = np.zeros(len(x))
        for peak in l:
            if peak[0] > Xmin and peak[0] < Xmax:
                y += eval(peakTypes[i])(x, peak[0], peak[1], width)

        if plotStyleList is None:
            ax.plot(x, y,'+-')
        else:
            ax.plot(x, y, plotStyleList[i])

    if labels is not None:
        ax.legend(labels, loc='best')

    plt.xlim(Xmin,Xmax)

    #Location of legend, use string or code
    #Best    0 upper right 1 upper left  2 lower left  3
    #lower right 4 Right   5 Center left 6 Center right    7
    #lower center    8 upper center    9 Center  10
    #if labels is not None:
        #ax.legend(labels, loc='lower right')

    #from scipy.stats import cauchy
    #yc = cauchy.pdf(x-1000)
    #ax.plot(x, yc, 'r', label='cauchy pdf')

    # Colors
    # ‘b’ Blue   ‘g’ Green ‘r’ Red  ‘b’ Blue  ‘c’ Cyan ‘m’ Magenta
    # ‘y’ Yellow ‘k’ Black ‘b’ Blue ‘w’ White

    # Markers
    # ‘.’ Point marker   ‘o’ Circle marker
    # ‘x’ X marker       ‘D’ Diamond marker
    # ‘H’ Hexagon marker ‘s’ Square marker ‘+’ Plus marker

    # Line Style
    # ‘-‘ Solid line   ‘—‘ Dashed line ‘-.’    Dash-dot line
    # ‘:’ Dotted line  ‘H’ Hexagon marker
    plt.show()
    return


def plotROAspectrum(peaksRaman, labelRaman, peakList, labels=None, Npts=1000, Xmin=None, Xmax=None,
    title='', width=15, peakType='Lorentzian', plotStyleList=None):

    if Xmin is None:
        Xmin = 1e6
        for l in peakList:
            tmp = max(min(l)[0] - 100,0)
            if tmp < Xmin: Xmin = tmp

    if Xmax is None:
        Xmax = 0
        for l in peakList:
            tmp = max(l)[0] + 100
            if tmp > Xmax: Xmax = tmp

    fig = plt.figure()

    ax=fig.add_axes([0.10,0.70,0.80,0.20])
    ax.set_title('Raman (top) and ROA (bottom) ' + title)
    ax.set_ylabel('Intensity')

    x = np.linspace(Xmin, Xmax, Npts)
    y = np.zeros(len(x))
    for p in peaksRaman:
        if p[0] > Xmin and p[0] < Xmax:
            y += eval(peakType)(x, p[0], p[1], width)

    if plotStyleList is None:
        ax.plot(x, y,'b.-')
    else:
        ax.plot(x, y, plotStyleList[0])

    ax.legend([labelRaman], loc='lower left')
    plt.xlim(Xmin,Xmax)

    ax2=fig.add_axes([0.10,0.10,0.80,0.55])
    ax2.set_xlabel('Wavenumber')
    ax2.set_ylabel('Intensity')

    for i,l in enumerate(peakList):
        y = np.zeros(len(x))
        for p in l:
            if p[0] > Xmin and p[0] < Xmax:
                y += eval(peakType)(x, p[0], p[1], width)

        if plotStyleList is None:
            ax2.plot(x, y,'.-')
        else:
            ax2.plot(x, y, plotStyleList[i+1])

        plt.xlim(Xmin,Xmax)

    if labels is not None:
        ax2.legend(labels, loc='best')

    plt.show()
    plt.clf()
    return

# ----------
# Compare spectra by numerically broadening them, and comparing
# the resulting curves.
def scaleMaxMagnitudeToOne(y):
    z = y / max(y)
    return z

def computeArea(peaks, Xmin, Xmax, Npts, width, peakType):
    """ Function to compute area. """
    x = np.linspace(Xmin, Xmax, Npts)
    y = roa.discretizedSpectrum(x, peaks, width, peakType)
    return np.trapz(y,x)

def computeAbsArea(peaks, Xmin, Xmax, Npts, width, peakType):
    """ Function to compute area of absolute difference between two spectra. """
    x = np.linspace(Xmin, Xmax, Npts)
    y = discretizedSpectrum(x, peaks, width, peakType)
    return np.trapz(abs(y),x)

def computeSquaredArea(peaks, Xmin, Xmax, Npts, width, peakType):
    """ Function to compute area of square of function """
    x = np.linspace(Xmin, Xmax, Npts)
    y = discretizedSpectrum(x, peaks, width, peakType)
    return np.trapz(y**2,x)

def computeDiffArea(peaks1, peaks2, Xmin, Xmax, Npts, width, peakType):
    """ Function to compute area of difference between two spectra. """
    x = np.linspace(Xmin, Xmax, Npts)
    y1 = discretizedSpectrum(x, peaks1, width, peakType)
    y2 = discretizedSpectrum(x, peaks2, width, peakType)
    return np.trapz(y2-y1,x)

def computeProductArea(peaks1, peaks2, Xmin, Xmax, Npts, width, peakType):
    """ Function to compute area of difference between two spectra. """
    x = np.linspace(Xmin, Xmax, Npts)
    y1 = discretizedSpectrum(x, peaks1, width, peakType)
    y2 = discretizedSpectrum(x, peaks2, width, peakType)
    return np.trapz(y1*y2,x)

def computeAbsDiffArea(peaks1, peaks2, Xmin, Xmax, Npts, width, peakType):
    """ Function to compute area of absolute difference between two spectra. """
    x = np.linspace(Xmin, Xmax, Npts)
    y1 = discretizedSpectrum(x, peaks1, width, peakType)
    y2 = discretizedSpectrum(x, peaks2, width, peakType)
    return np.trapz(abs(y2-y1),x)

# comparisonType:
# SNO, single-normalized overlap I(fs fr) / I(fr^2)                  -inf, +inf
# DNO, doubly-normalized overlap I(fs fr) / sqrt[ I(fs^2)*I(fr^2)]     -1, +1
# IDF, integrated difference function [I(fr^2) - I(fs^2)] / I(fr^2)  -inf, +1
# RADF,relative absolute difference function I[ |fs - fc| ] / I(fr)  -inf, +inf
def compareBroadenedSpectra(spRef, spTest, spectrumType=None, Npts=2000,
    Xmin=400, Xmax=4000, width=10, peakType='Lorentzian', comparisonType='RADF',
    symmetrize=False, printLevel=0):

    if comparisonType not in ['SNO','DNO','IDF','RADF']:
        raise('Unknown comparison type.')

    variableX = 'Frequency'
    if not spectrumType:
        spectrumType = 'ROA R-L Delta(180)'

    peaksRef  = list(zip(spRef.data[variableX], spRef.data[spectrumType]))
    peaksTest = list(zip(spTest.data[variableX], spTest.data[spectrumType]))

    if comparisonType == 'RADF':
        areaAbsRef  = computeAbsArea(peaksRef, Xmin, Xmax, Npts, width, peakType)
        areaAbsDiff = computeAbsDiffArea(peaksRef, peaksTest, Xmin, Xmax, Npts, width, peakType)
        tval = areaAbsDiff / areaAbsRef

        if symmetrize:
            areaAbsTest  = computeAbsArea(peaksTest, Xmin, Xmax, Npts, width, peakType)
            tval = 0.5 * (tval + areaAbsDiff / areaAbsTest)

        if printLevel:
            print(f'Area of Absolute Diff.         : {areaAbsDiff:10.4f}')
            print(f'Area of Absolute Reference     : {areaAbsRef:10.4f}')
            print(f'Relative Absolute Diff. (RADF) : {tval:10.4f}\n')
        return tval

    elif comparisonType == 'SNO':
        areaProduct = computeProductArea(peaksRef, peaksTest, Xmin, Xmax, Npts, width, peakType)
        areaSqrRef = computeSquaredArea(peaksRef, Xmin, Xmax, Npts, width, peakType)
        tval = areaProduct / areaSqrRef

        if symmetrize:
            areaSqrTest  = computeSquaredArea(peaksTest, Xmin, Xmax, Npts, width, peakType)
            tval = 0.5 * (tval + areaProduct / areaSqrTest)

        if printLevel:
            print(f'Area of Product                 : {areaProduct:10.4f}')
            print(f'Area of Ref^2                   : {areaSqrRef:10.4f}')
            print(f'Singly Normalized Overlap (SNO) : {tval:10.4f}\n')
        return tval

    elif comparisonType == 'DNO':
        areaSqrRef = computeSquaredArea(peaksRef, Xmin, Xmax, Npts, width, peakType)
        areaSqrTest = computeSquaredArea(peaksTest, Xmin, Xmax, Npts, width, peakType)
        areaProduct = computeProductArea(peaksRef, peaksTest, Xmin, Xmax, Npts, width, peakType)
        tval = areaProduct / np.sqrt(areaSqrRef * areaSqrTest)

        if printLevel:
            print(f'Area of Ref^2                  : {areaSqrRef:10.4f}')
            print(f'Area of Test^2                 : {areaSqrTest:10.4f}')
            print(f'Area of Product                : {areaProduct:10.4f}')
            print(f'Doubly Normalized Overlap (DNO): {tval:10.4f}\n')
        return tval

    elif comparisonType == 'IDF':
        areaSqrRef  = computeSquaredArea(peaksRef, Xmin, Xmax, Npts, width, peakType)
        areaSqrTest = computeSquaredArea(peaksTest, Xmin, Xmax, Npts, width, peakType)
        tval = (areaSqrRef-areaSqrTest)/areaSqrRef

        if symmetrize:
            tval = 0.5 * (tval + (areaSqrTest-areaSqrRef)/areaSqrTest)

        if printLevel:
            print(f'Area of Ref^2                  : {areaSqrRef:10.4f}')
            print(f'Area of Test^2                 : {areaSqrTest:10.4f}')
            print(f'Integrated Diff. Function (IDF): {tval:10.4f}\n')
        return tval
    else:
        raise("Could not compare.")
    return


def compareBroadenedSpectraFiles(fileRef, filesToTest, spectraToTest, Xmin=100,
    Xmax=4000, Npts=1000, peakType='Lorentzian', width=15, comparisonType='RADF'):

    print('Loading Reference Spectrum:')
    spRef = SPECTRUM.fromDictionaryOutputFile(fileRef)
    print(spRef)

    print('Loading Test Spectra:')
    spTest = {}
    for f in filesToTest:
        spTest[f] = SPECTRUM.fromDictionaryOutputFile(f)

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


