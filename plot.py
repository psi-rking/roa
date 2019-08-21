import numpy as np
import matplotlib.pyplot as plt
from math import pi,sqrt,exp

# Gaussian for adding width
# x:     cm^-1 at which to compute
# xzero: center of line
# R:     strength/intensity/area of line
# delta: half-width at 1/e * peak height
def Gaussian(x, xzero, R, Delta):
    y = x.copy()
    y[:] = [R / (Delta * sqrt(pi)) * exp(-((t-xzero)/Delta)**2) for t in x]
    return y

# Lorentzian for adding width
# x:     cm^-1 at which to compute
# xzero: center of line
# R:     strength/intensity/area of line
# gamma: half-width at 1/2 * peak height
def Lorentzian(x, xzero, R, Gamma):
    y = x.copy()
    y[:] = [R / pi * Gamma / ((t-xzero)**2 + Gamma**2) for t in x]
    return y

# peakList: tuples of (frequency, intensity)
# delta: linewidth parameter (perhaps in cm^-1)
# broadenList: list of booleans to indicate whether broadening is desired
# peakType: may be Lorentzian or Gaussian
def plotSpectrum(peakList, labels=None, Npoints=200, Xmin=None, Xmax=None,
plotStyleList=None, broadenList=None, title="Spectrum",
delta=15, peakType='Lorentzian'):

    if Xmin is None:
        Xmin = 1e6
        for l in peakList:
            tmp = max(min(l)[0] - 200,0)
            if tmp < Xmin: Xmin = tmp

    if Xmax is None:
        Xmax = 0
        for l in peakList:
            tmp = max(l)[0] + 200
            if tmp > Xmax: Xmax = tmp

    if broadenList is None:
        broadenList = [True] * len(peakList)

    fig = plt.figure()
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
        if broadenList[i]:
            y = 0
            x = np.linspace(Xmin, Xmax, Npoints)
            for peak in l:
                y += eval(peakType)(x, peak[0], peak[1], delta)
        else:
            x = []
            y = []
            for peak in l:
                if peak[0] > Xmin and peak[0] < Xmax:
                    x.append(peak[0])
                    y.append(peak[1])

        if plotStyleList is None:
            ax.plot(x, y,'+-')
        else:
            ax.plot(x, y, plotStyleList[i])

    if labels is not None:
        ax.legend(labels, loc='best')

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


def plotROAspectrum(peaksRaman, labelRaman, peakList, labels=None, Npoints=200, Xmin=None, Xmax=None,
    title='', delta=15, peakType='Lorentzian'):

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

    ax=fig.add_axes([0.10,0.50,0.80,0.40])
    ax.set_title('Raman (top) and ROA (bottom) ' + title)
    ax.set_ylabel('Intensity')

    x = np.linspace(Xmin, Xmax, Npoints)
    y = 0
    for p in peaksRaman:
        y += eval(peakType)(x, p[0], p[1], delta)
    ax.plot(x, y,'.-')
    plt.xlim(Xmin,Xmax)
    #ax.legend(labelRaman, loc='top left')

    ax2=fig.add_axes([0.10,0.10,0.80,0.40])
    ax2.set_xlabel('Wavenumber')
    ax2.set_ylabel('Intensity')

    for l in peakList:
        y = 0
        for p in l:
            y += eval(peakType)(x, p[0], p[1], delta)
        ax2.plot(x, y,'.-')
        plt.xlim(Xmin,Xmax)

    if labels is not None:
        ax2.legend(labels, loc='upper left')

    plt.show()
    plt.clf()

