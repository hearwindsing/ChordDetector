import numpy as np
import sklearn
from sklearn import preprocessing,metrics
from sklearn.naive_bayes import GaussianNB
from sklearn import cross_validation
from sklearn.cross_validation import KFold
from numpy import mean

import csv
import os

chroma = []
chords = []

def exportCollection(file):
    with open(file, 'rb') as csvfile:
        f = csv.reader(csvfile, delimiter=',')
        for row in f:
            if (row!='---')and(len(row)!=1):
                A = []
                for i in range(1,25):
                   A.append(float(row[i]))
                chroma.append(A)
                if len(row) == 26:
                    chords.append(row[25])
                else:
                    chords.append(row[25]+','+row[26])

exportCollection('chroma.txt')
print chords
                                                                                                                                                                                                                                                                                                        
print('Ya tut')
X = chroma
y = chords
lenChroma = len(chroma)
print lenChroma
kf = KFold(lenChroma, n_folds = 10)
model = GaussianNB()
print(model)

scores = sklearn.cross_validation.cross_val_score(estimator=model, X=X, y=y, cv=kf, scoring="accuracy")
score = mean(scores)
print score
