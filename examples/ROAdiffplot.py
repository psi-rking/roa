# Example of comparing two spectra.

import roa
import numpy as np
import matplotlib.pyplot as plt

sp = roa.SPECTRUM()
sp.readDictionaryOutputFile('6-31++Gs-cc2.sp.out')

sp2 = roa.SPECTRUM()
sp2.readPsi4OutputFile('lpoldl-cc2.out')

title = "HOOH basis set comparison"
labels = ["6-31++G*","Sadlej-lPoldL"]

# Choose range
Xmin = 600
Xmax = 2000

# Choose modes desired; includes peaks near range
R = []
for Inu, nu in enumerate(sp.data['Frequency']):
    if nu > 0.9*Xmin and nu < 1.1*Xmax:
        R.append(Inu)

ROA_type = 'ROA R-L Delta(180)'

peaksRaman = [(sp.data['Frequency'][i], sp.data[ROA_type][i]) for i in R] 
peaksRaman2 = [(sp2.data['Frequency'][i], sp2.data[ROA_type][i]) for i in R] 

Npts = 1000
width = 15
peakType = 'Lorentzian'
plotStyleList=['.-k','-k']

#roa.plot.plotSpectrum([peaksRaman,peaksRaman2],
#labels, Npts, Xmin, Xmax, plotStyleList, title, width, peakType)

fig = plt.figure()
# example of saving to pdf file : plt.savefig('foo.pdf')

ax=fig.add_axes([0.15,0.15,0.70,0.70])
ax.set_title(title)
ax.set_xlabel('wavenumber')
ax.set_ylabel('intensity')

x = np.linspace(Xmin, Xmax, Npts)
y  = roa.discretizedSpectrum(x, peaksRaman, width, peakType)
y2 = roa.discretizedSpectrum(x, peaksRaman2, width, peakType)

ax.plot(x, y, plotStyleList[0])
ax.plot(x, y2, plotStyleList[1])

ax.fill_between(x, y, y2)

if labels is not None:
    ax.legend(labels, loc='best')

plt.xlim(Xmin,Xmax)
plt.show()

