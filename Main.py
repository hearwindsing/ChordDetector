import numpy as np
import sklearn
from sklearn import preprocessing,metrics
from sklearn.naive_bayes import GaussianNB
from sklearn import cross_validation
from sklearn.cross_validation import KFold
from numpy import mean

import csv
import os

def exportCollection(file):
    chroma = []
    chords = []
    with open(file, 'rb') as csvfile:
        f = csv.reader(csvfile, delimiter=',')
        for row in f:
            chrm = map(float, row[2:])
            chroma.append(chrm)
            chords.append(row[0])
    return (chords, chroma)


if __name__ == '__main__':
    print "Load collection"
    (chords, chroma) = exportCollection('chroma.txt')

    print len(chroma)

    X = chroma
    y = chords
    lenChroma = len(X)

    kf = KFold(lenChroma, n_folds=10)
    model = GaussianNB()

    scores = sklearn.cross_validation.cross_val_score(estimator=model, X=X, y=y, cv=kf, scoring="accuracy")
    score = mean(scores)
    print score
