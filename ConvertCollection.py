import numpy as np
from sklearn import preprocessing,metrics
from sklearn.neighbors import KNeighborsClassifier

import csv
import os
from sklearn.naive_bayes import GaussianNB
from sklearn import cross_validation
from sklearn.cross_validation import KFold
from numpy import mean

from collections import Counter

result = open('chroma.txt','w')
chroma = []
chords = []

def chromaF(file):
    with open(file, 'rb') as csvfile:
        bothchroma = csv.reader(csvfile, delimiter=',')
        mas = []
        for row in bothchroma:
          A = []
          for i in range(1,len(row)):
             A.append(float(row[i]))
          mas.append(A)
        csvfile.close()
        return mas

def chordsMedian(timeInt,chordOne,mas):
   lenTimeInt = len(timeInt)-1
   apMed = timeInt[0] + abs(timeInt[lenTimeInt] - timeInt[0])/2
   flag = True
   i = 0
   while flag:
       if timeInt[i] < apMed:
           i = i + 1
       else:
           i = i - 1
           med = timeInt[i]
           flag = False
   res = []
   for j in range(0,25):
       res.append(mas[i][j])
   chroma.append(res)
   chords.append(chordOne)


def chordsF(file,mas):
    timeInt = []
    chordInt = []
    lenChroma = len(mas) - 1
    i = 0
    with open(file,'rb') as labfile:
     f = csv.reader(labfile,delimiter = '\t')
     for line in f:
        if line != []:
          l = line
          time = float(l[1])
          while (mas[i][0] <= time)and (i < lenChroma):
            timeInt.append(mas[i][0])
            chordInt.append(mas[i])
            i = i + 1
          chordOne = l[2]
          if timeInt != []:
             chordsMedian(timeInt,chordOne,chordInt)
             timeInt = []
             chordInt = []
          else:
              print 'Opachki'
              break;
        else:
          #  chords.append('N')
            labfile.close()
            break;

urlChroma = "/home/nastya/PycharmProjects/ChordDetector/MainDir/chroma/"
urlChords = "/home/nastya/PycharmProjects/ChordDetector/MainDir/chords/"

pt = os.listdir(urlChroma)
st = os.listdir(urlChords)

for i in range(0,len(pt)):
#     chroma = []
#     chords = []
     bothchroma = urlChroma+pt[i]+'/bothchroma.csv'
     full = urlChords+st[i]+'/full.lab'
     mas = chromaF(bothchroma)
     chordsF(full,mas)
     lenChroma = len(chroma)
#     result.write(pt[i]+'\n')
#     for j in range(0,lenChroma):
#        for t in range(0,25):
#            result.write(str(chroma[j][t])+',')
#        result.write(chords[j]+'\n')
#     result.write('---'+'\n')
     print pt[i]
     if len(chroma) != len(chords):
        print "chroma=chords"
        print len(chroma)
        print len(chords)
        break

# res = {}
# for x in chords:
#     res[x] = 1

print len(Counter(chords))

# print len(res)

result.close()