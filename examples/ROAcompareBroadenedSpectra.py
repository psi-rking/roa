import roa
import numpy as np



fileRef = 'asp-cc-pvdz-cc2.sp.out'
filesToTest = [
  '3-21G-cc2.sp.out',
  '6-31+Gs-cc2.sp.out',
  'cc-pvdz-cc2.sp.out',
  '6-31++Gs-cc2.sp.out' ]
spectraToTest = [
# 'IR Intensity',
# 'Raman Intensity (circular)',
 'ROA R-L Delta(180)',
# 'ROA R-L Delta(90)_z',
# 'ROA R-L Delta(90)_x',
# 'ROA R-L Delta(0)'
]

roa.compareBroadenedSpectra(fileRef, filesToTest, spectraToTest)

