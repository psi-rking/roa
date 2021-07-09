import numpy as np
import roa

fileRef = 'sadlej-lpol-fl-cc2.sp.out'
filesToTest = [
'3-21G-cc2.sp.out', 'aug-cc-pvtz-cc2.sp.out', 'sadlej-lpol-dl-cc2.sp.out',
'6-311++G(3df,3pd)-cc2.sp.out', 'cc-pvdz-cc2.sp.out', 'sadlej-lpol-ds-cc2.sp.out',
'6-31++G*-cc2.sp.out', 'cc-pvtz-cc2.sp.out', 'sadlej-lpol-fl-cc2.sp.out',
'6-31+G*-cc2.sp.out', 'd-aug-cc-pvdz-cc2.sp.out', 'sadlej-lpol-fs-cc2.sp.out',
'asp-cc-pvdz-cc2.sp.out', 'd-aug-cc-pvtz-cc2.sp.out',
'aug-cc-pvdz-cc2.sp.out', 'orp-cc2.sp.out' ]

#roa.compareSpectraFiles(fileRef, filesToTest, okThreshold=0.3)

spectraToTest = [
# 'IR Intensity',
# 'Raman Intensity (circular)',
 'ROA R-L Delta(180)',
# 'ROA R-L Delta(90)_z',
# 'ROA R-L Delta(90)_x',
# 'ROA R-L Delta(0)'
]

roa.compareBroadenedSpectra(fileRef, filesToTest, spectraToTest)

