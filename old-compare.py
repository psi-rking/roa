import numpy as np
from . import spectrum

# This needs fixed up later
def reorder(refSp, testSp ):
    # frequency 1 and 2 are within 1 percent
    if abs((refSp.data['Frequency'][0] - refSp.data['Frequency'][1])/refSp.data['Frequency'][0]) < 0.01:
        # look for distinguishing property
        v1 = refSp.data['ROA alpha*G'][0]
        v2 = refSp.data['ROA alpha*G'][1]
        refRatio = v1 / v2 if v2 else 1000
        v1 = testSp.data['ROA alpha*G'][0]
        v2 = testSp.data['ROA alpha*G'][1]
        ratio1 = v1 / v2 if v2 else 1000
        ratio2 = v2 / v1 if v1 else 1000
        if abs(ratio2 - refRatio) < abs(ratio1 - refRatio):
            for k in testSp.data.keys():
                if type(testSp.data[k]) == np.ndarray:
                    tval = testSp.data[k][0]
                    testSp.data[k][0] = testSp.data[k][1]
                    testSp.data[k][1] = tval
    return

def compareOutputFiles(refFile, testFiles, propKeys=None, alignModes=False, okThreshold=0.50): 
    print("Reading files:")
    refSpectrum = spectrum.SPECTRUM()
    refSpectrum.readPsi4OutputFile(refFile)
    print(refSpectrum)

    if propKeys is None:
        propKeys = spectrum.defaultKeys

    testSpectra = []
    for tF in testFiles:
        sp = spectrum.SPECTRUM()
        sp.readPsi4OutputFile(tF)
        print(sp)
        testSpectra.append(sp)

    print("Comparing data:")
    compareSpectra(refSpectrum, testSpectra, propKeys, alignModes, okThreshold)
    return

def compareSpectraFiles(refFile, testFiles, propKeys=None, alignModes=False, okThreshold=0.50): 
    print("Reading files:")
    refSpectrum = spectrum.SPECTRUM()
    refSpectrum.readDictionaryOutputFile(refFile)
    print(refSpectrum)

    if propKeys is None:
        propKeys = spectrum.defaultKeys

    testSpectra = []
    for tF in testFiles:
        sp = spectrum.SPECTRUM()
        sp.readDictionaryOutputFile(tF)
        print(sp)
        testSpectra.append(sp)

    print("Comparing data:")
    compareSpectra(refSpectrum, testSpectra, propKeys, alignModes, okThreshold)
    return

# Sref  = reference spectrum
# Stest = test spectra
def compareSpectra(Sref, Stest, propKeys, alignModes, okThreshold):
    # Do a test sort of the first couple of modes to try to line them up
    if alignModes:
        for spec in Stest:
            reorder(Sref, spec)

    aveDev = np.zeros( (len(Stest),len(propKeys)) )
    RMSDev = np.zeros( (len(Stest),len(propKeys)) )
    aveRelAbsDev = np.zeros( (len(Stest),len(propKeys)) )
    lblDev = []
    nbfDev = []
    for i, spec in enumerate(Stest):
        aveDev[i,:] = spec.compareToWithAveDev(Sref,propKeys)
        RMSDev[i,:] = spec.compareToWithRMSDev(Sref,propKeys)
        aveRelAbsDev[i,:] = spec.compareToWithAveRelAbsDev(Sref,propKeys)
        lblDev.append(spec.filename )
        nbfDev.append(spec.data['Number of basis functions'])

    # Print deviations
    print('\n'+148*'-')
    s = "%20s" % ('NBF')
    for k in propKeys:
        if k[:4] == 'ROA ':
            s += '%16s' % k[4:]
        else:
            s += '%16s' % k
    print(s)
    print(148*'-')

    print("** Average Deviation from %s" % Sref.filename)
    s = ''
    for i in range(len(lblDev)):
        s += "%-16s%4d" % (lblDev[i], nbfDev[i]) # filename
        for p in range(len(propKeys)):
            s += "%16.4f" % aveDev[i,p]
        s += '\n'
    print(s)
    
    print("** RMS Deviation from %s" % Sref.filename)
    s = ''
    for i in range(len(lblDev)):
        s += "%-16s%4d" % (lblDev[i], nbfDev[i]) # filename
        for p in range(len(propKeys)):
            s += "%16.4f" % RMSDev[i,p]
        s += '\n'
    print(s)
    
    print("** Average Relative Absolute Deviation from %s" % Sref.filename)
    s = ''
    for i in range(len(lblDev)):
        s += "%-16s%4d" % (lblDev[i], nbfDev[i]) # filename
        for p in range(len(propKeys)):
            s += "%16.4f" % aveRelAbsDev[i,p]
        s += '\n'
    print(s)
    
    print("Number of modes compared: %d\n" % len(Sref.data['Frequency']))
    
    best = []
    for i in range(len(propKeys)):
        o = np.argsort( RMSDev[:,i] )
        v = [o[j] for j in range(len(lblDev))]
        best.append(v)

    fieldwidth = 25
    
    print("-" * fieldwidth * len(propKeys[0:4]))
    print("Ranked by RMS deviation from %s (%d). Best on top." %
        (Sref.filename, Sref.data['Number of basis functions']))
    print("Shown if average relative absolute deviation is < %4.2f." % okThreshold)
    s = ''
    for k in propKeys[0:4]:
        s += "{0:>{1}}".format(k,fieldwidth)
    print(s)
    print("-" * fieldwidth * len(propKeys[0:4]))
    
    for place in range(len(Stest)):
        s = ""
        for i, b in enumerate(best[0:4]):
           if aveRelAbsDev[b[place], i] < okThreshold:
                s += "{0:<{1}}".format(('(' + str(nbfDev[ b[place] ]) + ')' +
                             lblDev[b[place]])[0:24], fieldwidth)
           else:
               s += fieldwidth * " "
        print(s)
    print( "-" * fieldwidth * len(propKeys[0:4]))

    #print("Ranked by RMS deviation from %s" % refFile)
    #print("Shown if average relative absolute deviation is < %4.2f." % okThreshold)

    s = ''
    for k in propKeys[4:]:
        s += "{0:>{1}}".format(k,fieldwidth)
    print(s)
    print( "-" * fieldwidth * len(propKeys[4:]))

    # if you don't want part of file name in output
    #import re
    #rmstr = '-ccsd'

    for place in range(len(Stest)):
        s = ""
        for i, b in enumerate(best[4:]):
           if aveRelAbsDev[b[place], 4+i] < okThreshold:
               s += "{0:<{1}}".format(('(' + str(nbfDev[ b[place] ]) + ')' +
                            lblDev[b[place]])[0:24], fieldwidth)
           else:
               s += fieldwidth * " "
        print(s)
    print( "-" * fieldwidth * len(propKeys[4:]))

    return
