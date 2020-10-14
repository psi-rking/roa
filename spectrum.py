import numpy as np
from numpy import array

def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]

defaultKeys = [ 'Frequency', 'ROA alpha*G', 'ROA Beta(G)^2', 'ROA Beta(A)^2',
'ROA R-L Delta(90)_z', 'ROA R-L Delta(90)_x', 'ROA R-L Delta(0)', 'ROA R-L Delta(180)' ]
#['Raman Intensity (linear)'] 
#['Raman Intensity (circular)'
#['ROA alpha*G']                = alpha[0:last] * G[0:last]
#['ROA Beta(G)^2']              = beta_G2[0:last]
#['ROA Beta(A)^2']              = betaA2[0:last]

class SPECTRUM(object):

    def __init__(self, fileName=None):
        self.filename = fileName
        self.data = {}

    def __str__(self):
        s  = '\n ROA Diff. Parameter R-L (Ang^4/amu * 1000)'
        s += ' Filename: %15s,  # of b.f.: %3d \n' % (self.filename,self.data['Number of basis functions'])
        s += 132*'-' + '\n'
        s += '    '
        for key in defaultKeys:
            if key[:4] == 'ROA ':
                s += '%16s' % key[4:] # truncate off 'ROA '
            else:
                s += '%16s' % key
        s += '\n'

        s += 132*'-' + '\n'
        for i in range(len(self.data['Frequency'])):
            s += '%4d'    % (i+1)
            for key in defaultKeys:
                s += '%16.4f' % self.data[key][i]
            s += '\n'
        return s

    def print_all(self):
        # assume everything is either scalar or is an array of dimension 1 by # of frequencies
        keys = list(self.data.keys())
        fieldwidth = 25

        s = '** Scalar Quantities:\n'
        # Print scalars first
        for k in list(keys):
           if (not hasattr(self.data[k],'__len__')) or isinstance(self.data[k],str):
               s += '{:30}{:^30}\n'.format(k,self.data[k])
               keys.remove(k)

        s += '** Vector Quantities:\n'
        for keyRow in batch(keys,4):
            s += 103*'-' + '\n' + '   '
            for k in keyRow:
                s += '{:>{fw}s}'.format(k,fw=fieldwidth)
            s += '\n'
            s += 103*'-' + '\n'

            for i in range(len(self.data['Frequency'])):
                s += '{:3d}'.format(i+1)
                for k in keyRow:
                    s += '{:{fw}.4f}'.format(self.data[k][i],fw=fieldwidth)
                s += '\n'

        s += 103*'-' + '\n'
        print(s)
        return


    def readDictionaryOutputFile(self,filename):
        self.filename = filename
        with open(filename,'r') as f:
            self.data = eval( f.read() )
        if filename[-4:] == '.dat':
            filename = filename[:-4]
        filename = filename.replace("spectra","sp")
        self.filename = filename

    def writeDictionaryOutputFile(self, outfile):
        with open(outfile,'w') as f:
            f.write(str(self.data))

    def readPsi4OutputFile(self,filename):
        freq = []
        alphaG = []
        betaG2 = []
        betaA2 = []
        delta_z_90 = []
        delta_x_90 = []
        delta_0  = []
        delta_180  = []
        IRIntensity = []
        RamanLinear  = []
        RamanCircular  = []
        wfn = ''
        basis = ''
        nbf = 0

        with open(filename,'r') as f:
            startInvariants = 0
            startDifference = 0
            startParameters = 0
            startOldFrequencies = 0
            oldStyle = False
            toInvariantsLine = 0
            toDifferenceLine = 0
            toParametersLine = 0
            toStartOldFrequenciesLine = 0
        
            for line in f:
                words = line.split()
                cnt = 0
                if len(words) == 3:
                    if words[0] == 'Wavefunction' and words[1] == '=':
                        wfn = words[2]
                    elif words[0] == 'Basis' and words[1] == 'Set:':
                        basis = words[2]

                if len(words) > 4:
                    if words[0] == 'Number' and words[1] == 'of' and words[2] == 'basis':
                        nbf = int(words[4])
        
                if startInvariants:
                   toInvariantsLine += 1
                if startDifference:
                   toDifferenceLine += 1
                if startParameters:
                   toParametersLine += 1
                if startOldFrequencies:
                   toStartOldFrequenciesLine += 1

                if startParameters == False:
                    if len(words) > 2:
                        # OLD style output from 2019
                        if words[0] == "Harmonic" and words[1] == "Freq." and words[2] == "IR":
                            print("Reading old 2019 style output")
                            startOldFrequencies = True
                            oldStyle = True
                        elif words[0] == 'Raman' and words[1] == 'Scattering' and words[2] == 'Parameters':
                            startParameters = True

                if startOldFrequencies and toStartOldFrequenciesLine > 2:
                    if len(words) < 2: # all done
                        startOldFrequencies = False
                    elif words[1][-1] == 'i': # hit an imaginary frequency
                        startOldFrequencies = False
                    elif float(words[1]) < 8.0: # hit a rotation
                        startOldFrequencies = False
                    else:
                       #freq.append(float(words[1]))
                       IRIntensity.append(float(words[2]))

                lineOffset = (4 if oldStyle else 5)
                if startParameters and toParametersLine > lineOffset:
                    if len(words) < 2: # all done
                        startParameters = False
                    elif words[1][-1] == 'i': # hit an imaginary frequency
                        startParameters = False
                    elif float(words[1]) < 8.0: # hit a rotation
                        startParameters = False
                    else:
                       if not oldStyle:
                           IRIntensity.append(float(words[2]))
                           RamanLinear.append(float(words[6]))
                           RamanCircular.append(float(words[8]))
                       else:
                           RamanLinear.append(float(words[4]))
                           RamanCircular.append(float(words[6]))
        
                if startInvariants == False:
                    if len(words) > 2:
                        if words[0] == 'ROA' and words[1] == 'Scattering' and words[2] == 'Invariants':
                            startInvariants = True

                lineOffset = (4 if oldStyle else 3)
                if startInvariants and toInvariantsLine > lineOffset:
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
            self.data['IR Intensity']       = np.array( IRIntensity )
            self.data['Raman Intensity (linear)']  = np.array( RamanLinear )
            self.data['Raman Intensity (circular)'] = np.array( RamanCircular )
            self.data['Calculation Type']   = wfn + '/' + basis
            self.data['Number of basis functions'] = nbf
        if filename[-4:] in ['.out', '.dat']:
            self.filename = filename[:-4]
        else:
            self.filename = filename

    def compareToWithAveDev(s, o, keys=None):  #self,other
        if len(s.data['Frequency']) != len(o.data['Frequency']):
            raise Exception('Spectra cannot be compared')

        if keys is None:
            keys = defaultKeys

        diff = np.zeros( len(keys) )
        for i, k in enumerate(keys):
            diff[i] = np.mean(s.data[k] - o.data[k])

        return diff

    def compareToWithRMSDev(s, o, keys=None):  #self,other
        if len(s.data['Frequency']) != len(o.data['Frequency']):
            raise('spectra cannot be compared')

        if keys is None:
            keys = defaultKeys

        diff = np.zeros( len(keys) )
        for i, k in enumerate(keys):
            diff[i] = np.sqrt(np.mean( (s.data[k] - o.data[k])**2 ) )

        return diff
        
    # omit values if reference is < 3.5% of maximum value
    def compareToWithRelDev(s, o, keys=None, omitBelow=0.035):  #self,other
        if len(s.data['Frequency']) != len(o.data['Frequency']):
            raise('spectra cannot be compared')

        if keys is None:
            keys = defaultKeys

        diff = np.zeros( len(keys) )
        for i, k in enumerate(keys):
            diff[i] = aveRelDev(s.data[k], o.data[k], omitBelow)
        return diff

    def compareToWithAveRelAbsDev(s, o, keys=None, omitBelow=0.035):  #self,other
        if len(s.data['Frequency']) != len(o.data['Frequency']):
            raise('spectra cannot be compared')

        if keys is None:
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
        else: print('aveRelDev: ignoring mode %d' % i)

    rval /= num
    return rval

def aveRelAbsDev(Values, refValues, omitBelow):
    maxValCounted = omitBelow*max(refValues)
    num = 0
    rval = 0.0
    ignore_mode = []
    for i in range(len(Values)):
        if abs(refValues[i]) > maxValCounted:
            num += 1
            rval += abs( (Values[i] - refValues[i]) / refValues[i])
        else:
            ignore_mode.append(i)

    if len(ignore_mode) != 0:
        print('In aveRelAbsDev, ignoring modes with small ref. values:', [m+1 for m in ignore_mode])
    rval /= num
    return rval

