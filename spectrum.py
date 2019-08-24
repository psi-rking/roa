import numpy as np
from numpy import array

defaultKeys = [ 'Frequency', 'ROA alpha*G', 'ROA Beta(G)^2', 'ROA Beta(A)^2',
'ROA R-L Delta(90)_z', 'ROA R-L Delta(90)_x', 'ROA R-L Delta(0)', 'ROA R-L Delta(180)' ]
#['Raman Intensity (linear)'] 
#['Raman Intensity (circular)'
#['ROA alpha*G']                = alpha[0:last] * G[0:last]
#['ROA Beta(G)^2']              = beta_G2[0:last]
#['ROA Beta(A)^2']              = betaA2[0:last]

class SPECTRUM(object):

    def __init__(self):
        self.filename = None
        self.data = {}
        #if filename is not None:
        #    self.readOutput()

    def __str__(self):
        s  = '   ROA Diff. Parameter R-L (Ang^4/amu * 1000)       '
        s += '  Filename: %20s  # of b.f.: %3d \n' % (self.filename,self.data['Number of basis functions'])
        s += '----------------------------------------------------- '
        s += '----------------------------------------------------\n'
        s += '    '
        for key in defaultKeys:
            s += '%12s' % key
        s += '\n'
        #s += '     Harmonic Freq.  AlphaG   Beta(G)^2    Beta(A)^2   '
        #s += 'Delta_z(90)  Delta_x(90)   Delta(0)   Delta(180)   \n'

        s += '----------------------------------------------------- '
        s += '----------------------------------------------------\n'

        for i in range(len(self.data['Frequency'])):
            s += '%4d'    % (i+1)
            for key in defaultKeys:
                s += '%12.4f' % self.data[key][i]
            s += '\n'
        return s

    def readDictionaryOutputFile(self,filename):
        self.filename = filename
        with open(filename,'r') as f:
            self.data = eval( f.read() )

    def writeDictionaryOutputFile(self, outfile):
        with open(outfile,'w') as f:
            f.write(str(self.data))

    def readPsi4OutputFile(self,filename):
        self.filename = filename
        freq = []
        alphaG = []
        betaG2 = []
        betaA2 = []
        delta_z_90 = []
        delta_x_90 = []
        delta_0  = []
        delta_180  = []
        RamanLinear  = []
        RamanCircular  = []

        with open(filename,'r') as f:
            startInvariants = 0
            startDifference = 0
            startParameters = 0
            toInvariantsLine = 0
            toDifferenceLine = 0
            toParametersLine = 0
        
            for line in f:
                words = line.split()
                cnt = 0

                if len(words) > 4:
                    if words[0] == 'Number' and words[1] == 'of' and words[2] == 'basis':
                        self.data['Number of basis functions'] = int(words[4])
        
                if startInvariants:
                   toInvariantsLine += 1
                if startDifference:
                   toDifferenceLine += 1
                if startParameters:
                   toParametersLine += 1

                if startParameters == False:
                    if len(words) > 2:
                        if words[0] == 'Raman' and words[1] == 'Scattering' and words[2] == 'Parameters':
                            startParameters = True

                if startParameters and toParametersLine > 4:
                    if len(words) < 2: # all done
                        startParameters = False
                    elif words[1][-1] == 'i': # hit an imaginary frequency
                        startParameters = False
                    elif float(words[1]) < 8.0: # hit a rotation
                        startParameters = False
                    else:
                       RamanLinear.append(float(words[4]))
                       RamanCircular.append(float(words[6]))
        
                if startInvariants == False:
                    if len(words) > 2:
                        if words[0] == 'ROA' and words[1] == 'Scattering' and words[2] == 'Invariants':
                            startInvariants = True
        
                if startInvariants and toInvariantsLine > 4:
                    if len(words) < 2: # all done
                        startInvariants = False
                    elif words[1][-1] == 'i': # hit an imaginary frequency
                        startInvariants = False
                    elif float(words[1]) < 8.0: # hit a rotation
                        startInvariants = False
                    else:
                       freq.append(  float(words[1]))
                       alphaG.append(float(words[2]))
                       betaG2.append(float(words[3]))
                       betaA2.append(float(words[4]))
        
                if startDifference == False:
                    if len(words) > 2:
                        if words[0] == 'ROA' and words[1] == 'Difference' and words[2] == 'Parameter':
                            startDifference = True
        
                if startDifference and toDifferenceLine > 4:
                    if len(words) < 2: # all done
                        startDifference = False
                    elif words[1][-1] == 'i': # hit an imaginary frequency
                        startDifference = False
                    elif float(words[1]) < 8.0:
                        startDifference = False
                    else:
                       delta_z_90.append( float(words[2]))
                       delta_x_90.append( float(words[3]))
                       delta_0.append(    float(words[4]))
                       delta_180.append(  float(words[5]))

            self.data['Frequency']          = np.array( freq )
            self.data['ROA alpha*G']        = np.array( alphaG )
            self.data['ROA Beta(G)^2']      = np.array( betaG2 )
            self.data['ROA Beta(A)^2']      = np.array( betaA2 )
            self.data['ROA R-L Delta(90)_z']= np.array( delta_z_90 )
            self.data['ROA R-L Delta(90)_x']= np.array( delta_x_90 )
            self.data['ROA R-L Delta(0)']   = np.array( delta_0 )
            self.data['ROA R-L Delta(180)'] = np.array( delta_180 )
            #['IR Intensity']              TODO
            self.data['Raman Intensity (linear)']  = np.array( RamanLinear )
            self.data['Raman Intensity (circular)'] = np.array( RamanCircular )

    def compareToWithAveDev(s, o, keys=None):  #self,other
        if len(s.data['Frequency']) != len(o.data['Frequency']):
            raise Exception('Spectra cannot be compared')

        if keys == None:
            keys = defaultKeys

        diff = np.zeros( len(keys) )
        for i, k in enumerate(keys):
            diff[i] = np.mean(s.data[k] - o.data[k])

        return diff

    def compareToWithRMSDev(s, o, keys=None):  #self,other
        if len(s.data['Frequency']) != len(o.data['Frequency']):
            raise('spectra cannot be compared')

        if keys == None:
            keys = defaultKeys

        diff = np.zeros( len(keys) )
        for i, k in enumerate(keys):
            diff[i] = np.sqrt(np.mean( (s.data[k] - o.data[k])**2 ) )

        return diff
        
    # omit values if reference is < 3.5% of maximum value
    def compareToWithRelDev(s, o, keys=None, omitBelow=0.035):  #self,other
        if len(s.data['Frequency']) != len(o.data['Frequency']):
            raise('spectra cannot be compared')

        if keys == None:
            keys = defaultKeys

        diff = np.zeros( len(keys) )
        for i, k in enumerate(keys):
            diff[i] = aveRelDev(s.data[k], o.data[k], omitBelow)
        return diff

    def compareToWithAveRelAbsDev(s, o, keys=None, omitBelow=0.035):  #self,other
        if len(s.data['Frequency']) != len(o.data['Frequency']):
            raise('spectra cannot be compared')

        if keys == None:
            keys = defaultKeys

        diff = np.zeros( len(keys) )
        for i, k in enumerate(keys):
            diff[i] = aveRelAbsDev(s.data[k], o.data[k], omitBelow)
        return diff

        
def aveRelDev(Values, refValues, omitBelow):
    maxValCounted = omitBelow*max(refValues)
    num = 0
    rval = 0.0
    for i in range(len(Values)):
        if abs(refValues[i]) > maxValCounted:
            num += 1
            rval += (Values[i] - refValues[i]) / refValues[i]
        else: print('ignoring mode %d' % i)

    rval /= num
    return rval

def aveRelAbsDev(Values, refValues, omitBelow):
    maxValCounted = omitBelow*max(refValues)
    num = 0
    rval = 0.0
    for i in range(len(Values)):
        if abs(refValues[i]) > maxValCounted:
            num += 1
            rval += abs( (Values[i] - refValues[i]) / refValues[i])
        else:
            print('ignore in aveRelAbsDev mode %d, value %10.5f, maxvalue %10.5f' % (i, refValues[i], max(refValues)))

    rval /= num
    return rval

