import numpy as np
import roa

fileRef = 'lpolfl-cc2.out'
filesToTest = [
  'lpolfs-cc2.out',
  'lpoldl-cc2.out',
  'lpolds-cc2.out',
 ]

roa.compareOutputFiles(fileRef, filesToTest, okThreshold=0.3)

