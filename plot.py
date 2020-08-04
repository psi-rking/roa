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
# delta: half-width at 1/e * peak height
def Gaussian(x, xzero, R, Delta):
    xzero = find_nearest(x, xzero)  # make sure our finite representation captures peak
    y = np.zeros(len(x))
    y[:] = [R / (Delta * sqrt(pi)) * exp(-((t-xzero)/Delta)**2) for t in x]
    return y

# Lorentzian for adding width
# x:     cm^-1 at which to compute
# xzero: center of line
# R:     strength/intensity/area of line
# gamma: half-width at 1/2 * peak height
def Lorentzian(x, xzero, R, Gamma):
    xzero = find_nearest(x, xzero)  # make sure our finite representation captures peak
    y = np.zeros(len(x))
    y[:] = [R / pi * Gamma / ((t-xzero)**2 + Gamma**2) for t in x]
    return y

def DeltaPeak(x, xzero, R, Delta=None):
    y = np.zeros(len(x))
    y[find_nearest_idx(x, xzero)] = R
    return y

# x = pts at which evaluation should be done
def discretizedSpectrum(x, peaks, delta=15, peakType='Lorentzian'):
    Xmin = min(x)
    Xmax = max(x)
    y = np.zeros(len(x))
    for p in peaks:
        if p[0] > Xmin and p[0] < Xmax:
            y += eval(peakType)(x, p[0], p[1], delta)
    return y

# peakList: tuples of (frequency, intensity)
# delta: linewidth parameter (perhaps in cm^-1)
# peakType: may be Lorentzian or Gaussian or DeltaPeak; may be list
def plotSpectrum(peakList, labels=None, Npoints=200, Xmin=None, Xmax=None,
plotStyleList=None, title="Spectrum", delta=15, peakType='Lorentzian'):

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
        x = np.linspace(Xmin, Xmax, Npoints)
        y = np.zeros(len(x))
        for peak in l:
            if peak[0] > Xmin and peak[0] < Xmax:
                y += eval(peakTypes[i])(x, peak[0], peak[1], delta)

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


def plotROAspectrum(peaksRaman, labelRaman, peakList, labels=None, Npoints=200, Xmin=None, Xmax=None,
    title='', delta=15, peakType='Lorentzian', plotStyleList=None):

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

    x = np.linspace(Xmin, Xmax, Npoints)
    y = np.zeros(len(x))
    for p in peaksRaman:
        if p[0] > Xmin and p[0] < Xmax:
            y += eval(peakType)(x, p[0], p[1], delta)

    if plotStyleList is None:
        ax.plot(x, y,'b.-')
    else:
        ax.plot(x, y, plotStyleList[0])

    plt.xlim(Xmin,Xmax)
    #ax.legend(labelRaman, loc='lower left')

    ax2=fig.add_axes([0.10,0.10,0.80,0.55])
    ax2.set_xlabel('Wavenumber')
    ax2.set_ylabel('Intensity')

    for i,l in enumerate(peakList):
        y = np.zeros(len(x))
        for p in l:
            if p[0] > Xmin and p[0] < Xmax:
                y += eval(peakType)(x, p[0], p[1], delta)

        if plotStyleList is None:
            ax2.plot(x, y,'.-')
        else:
            ax2.plot(x, y, plotStyleList[i+1])

        plt.xlim(Xmin,Xmax)

    if labels is not None:
        ax2.legend(labels, loc='best')
    #    ax2.legend(labels, loc='upper left')

    plt.show()
    plt.clf()

