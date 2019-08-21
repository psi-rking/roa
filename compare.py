import numpy as np
from . import spectrum

#defaultKeys = [ 'Frequency', 'ROA alpha*G', 'ROA Beta(G)^2', 'ROA Beta(A)^2',
#'ROA R-L Delta(90)_z', 'ROA R-L Delta(90)_x', 'ROA R-L Delta(0)', 'ROA R-L Delta(180)' ]

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

def compareFiles(refFile, testFiles, propKeys=None, alignModes=False, okThreshold=0.20): 

    ref = spectrum.SPECTRUM()
    ref.readPsi4OutputFile(refFile)
    print(ref)

    if propKeys is None:
        propKeys = spectrum.defaultKeys

    testSpectra = []
    for name in testFiles:
        sp = spectrum.SPECTRUM()
        sp.readPsi4OutputFile(name)
        testSpectra.append(sp)

    # Do a test sort of the first couple of modes to try to line them up
    if alignModes:
        for sp in testSpectra:
            reorder(ref, sp)

    aveDev = np.zeros( (len(testSpectra),len(propKeys)) )
    RMSDev = np.zeros( (len(testSpectra),len(propKeys)) )
    aveRelAbsDev = np.zeros( (len(testSpectra),len(propKeys)) )
    lblDev = []
    nbfDev = []
    for i, sp in enumerate(testSpectra):
        aveDev[i,:] = sp.compareToWithAveDev(ref,propKeys)
        RMSDev[i,:] = sp.compareToWithRMSDev(ref,propKeys)
        aveRelAbsDev[i,:] = sp.compareToWithAveRelAbsDev(ref,propKeys)
        lblDev.append(sp.filename[:-4] )
        nbfDev.append(sp.data['NBF'])

    # Print deviations
    s = "%20s%4s" % ('','NBF')
    for k in propKeys:
        s += '%12s' % k
    print(s)

    print("** Average Deviation from %s" % refFile)
    s = ''
    for i in range(len(lblDev)):
        s += "%-20s%4d" % (lblDev[i], nbfDev[i]) # filename
        for p in range(len(propKeys)):
            s += "%12f" % aveDev[i,p]
        s += '\n'
    print(s)
    
    print("** RMS Deviation from %s" % refFile)
    s = ''
    for i in range(len(lblDev)):
        s += "%-20s%4d" % (lblDev[i], nbfDev[i]) # filename
        for p in range(len(propKeys)):
            s += "%12f" % RMSDev[i,p]
        s += '\n'
    print(s)
    
    print("** Average Relative Absolute Deviation from %s" % refFile)
    s = ''
    for i in range(len(lblDev)):
        s += "%-20s%4d" % (lblDev[i], nbfDev[i]) # filename
        for p in range(len(propKeys)):
            s += "%12f" % aveRelAbsDev[i,p]
        s += '\n'
    print(s)
    
    print("Number of modes compared: %d\n" % len(ref.data['Frequency']))
    
    best = []
    for i in range(len(propKeys)):
        o = np.argsort( RMSDev[:,i] )
        v = [o[j] for j in range(len(lblDev))]
        best.append(v)

    fieldwidth = 25
    
    print("_" * fieldwidth * len(propKeys[0:4]))
    print("Ranked by RMS deviation from %s" % refFile)
    print("Shown if average relative absolute deviation is < %4.2f." % okThreshold)
    s = ''
    for k in propKeys[0:4]:
        s += "{0:^{1}}".format(k,fieldwidth)
    print(s)
    print("_" * fieldwidth * len(propKeys[0:4]))
    
    for place in range(len(testFiles)):
        s = ""
        for i, b in enumerate(best[0:4]):
           if aveRelAbsDev[b[place], i] < okThreshold:
                s += "{0:>{1}}".format(lblDev[ b[place] ] + '(' + str(nbfDev[ b[place] ]) +')', fieldwidth)
           else:
               s += fieldwidth * " "
        print(s)
    print( "_" * fieldwidth * len(propKeys[0:4]))

    print("Ranked by RMS deviation from %s" % refFile)
    print("Shown if average relative absolute deviation is < %4.2f." % okThreshold)

    s = ''
    for k in propKeys[4:]:
        s += "{0:^{1}}".format(k,fieldwidth)
    print(s)
    print( "_" * fieldwidth * len(propKeys[4:]))

    for place in range(len(testFiles)):
        s = ""
        for i, b in enumerate(best[4:]):
           if aveRelAbsDev[b[place], 4+i] < okThreshold:
                s += "{0:>{1}}".format(lblDev[ b[place] ] + '(' + str(nbfDev[ b[place] ]) +')', fieldwidth)
           else:
               s += fieldwidth * " "
        print(s)
    print( "_" * fieldwidth * len(propKeys[4:]))

    return
