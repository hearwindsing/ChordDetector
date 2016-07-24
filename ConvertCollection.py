import itertools

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

result = open('ch.txt','w')
chroma = []
chords = []

def takeMedian(chromas):
    median = len(chromas) / 2
    return [chromas[median]]

# define is chord ended: maj, min, 7, min7, maj7
def findChords(str):
    if str == 'N' or str == 'X':
       return False
    # return True
    res  = str.split(':')
    if len(res) < 2:
        print str
    return res[1] in {'maj', 'min', '7', 'maj7', 'min7'}

# Read .lab file
# Returns list of tuples (start, end, chord)
def loadChord(chordFile):
    chords = []
    with open(chordFile, 'r') as csvfile:
        csvreader = csv.reader(csvfile, delimiter='\t')
        for row in csvreader:
            if row != []:
                start = float(row[0])
                end = float(row[1])
                chrd = row[2].replace(',', ';')
                if findChords(chrd):
                    chords.append((start, end, chrd))
    return chords

# Read bothchroma.csv file
# Returns list of 24-dim chroma vectors
def loadChroma(chromaFile):
    chroma = []
    with open(chromaFile, 'r') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',')
        for row in csvreader:
            if row != []:
                chrm = map(float, row[1:])
                chroma.append(chrm)
    return chroma

# Joins tuple (start, end, chord) with corresponded chroma
# Returns list of tuples ( chord, time, chroma )
# Param `func` defines function that takes list of chromas and should return single chroma vector
def joinChordsChroma(chords, chroma, func):
    currChroma = chroma
    res = []
    for (timeStart, timeEnd, chrd) in chords:
        tmp = itertools.dropwhile(lambda v: v[0] < timeStart, currChroma)
        corrChroma = list(itertools.takewhile(lambda v: v[0] <= timeEnd, tmp))

        if len(corrChroma) > 0:
            chosenChroma = func(corrChroma)
            for chrm in chosenChroma:
                res.append( [chrd] + chrm )
            currChroma = currChroma[len(corrChroma):]
        else:
            print "Empty chromas: start={}, end={}".format(timeStart, timeEnd)
            currChroma =  currChroma[1:]
    return res

def convertData(urlChroma, urlChords, outFile):
    pt = os.listdir(urlChroma)
    st = os.listdir(urlChords)

    with open(outFile, 'w') as file:
        for i in range(0, len(pt) / 4):
            bothchroma = os.path.join(urlChroma, pt[i], 'bothchroma.csv')
            full = os.path.join(urlChords, st[i], 'full.lab')

            chroma = loadChroma(bothchroma)
            chords = loadChord(full)
            joined = joinChordsChroma(chords, chroma, takeMedian)
#lambda x: x
            for row in joined:
                strRow = ", ".join([pt[i]] + map(str, row))
                file.write(strRow + '\n')

            if pt[i] != st[i]:
                print "Error"

            print pt[i]

            # if len(chroma) != len(chords):
            #    print "chroma != chords"
            #    print len(chroma)
            #    print len(chords)
            #    break

if __name__ == '__main__':

    urlChroma = "./chroma/"
    urlChords = "./chords/"
    outFile = "chroma.txt"

    convertData(urlChroma, urlChords, outFile)
