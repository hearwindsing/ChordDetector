import gc
import csv
import itertools

import numpy as np
import pandas as pd

from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn import linear_model
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import make_scorer, accuracy_score, confusion_matrix

# 0-column - song_id
# 1-column - chord
# 2-column - time
# 3:- columns - chromas
SONG_ID_COL = 0
CHORD_COL = 1
TIME_COL = 2
CHROMA_START_COL = 3

result = open('result.txt','w')

def takeMedian(chromas):
    if not chromas:
        return []
    median = len(chromas) / 2
    return [chromas[median]]



def filterChordsNotNX(str):
    str = str.strip()
    if not str or str == 'N' or str == 'X':
       return False
    return True

# define is chord ended: maj, min, 7, min7, maj7
def filterChords5(str):
    str = str.strip()
    if not filterChordsNotNX(str):
        return False
    res  = str.split(':')
    # if len(res) < 2:
    #     print str
    return res[1] in {'maj', 'min', '7', 'maj7', 'min7'}

def getDataset(files, chordFilter, takeFunc):
    res = []
    takeChromas = takeFunc if takeFunc else lambda x: x
    for f in files:
        # print '...'
        with open(f, 'rb') as fd:
            reader = csv.reader(fd, delimiter=',')
            while True:
                try:
                    chrd = ''
                    while not chordFilter(chrd):
                        first = next(reader)
                        chrd = first[CHORD_COL]
                    group = [first] + list(itertools.takewhile(lambda row: row[CHORD_COL] == chrd, reader))
                    chosen = takeChromas(group)
                    data = []
                    for row in chosen:
                        data.append(row[:CHROMA_START_COL] + map(float, row[CHROMA_START_COL:]))
                    res += data
                except StopIteration:
                    break
    return pd.DataFrame(res)

def getDataset10(files, chordFilter, takeFunc):
    res = []
    for f in files:
        # print '...'
        with open(f, 'rb') as fd:
            reader = csv.reader(fd, delimiter=',')
            n = 0
            for row in reader:
                chrd = row[CHORD_COL]
                if chordFilter(chrd):
                    if n % 10 == 0:
                        res.append(row[:CHROMA_START_COL] + map(float, row[CHROMA_START_COL:]))
                    n += 1
    return pd.DataFrame(res)


# returns train dataset which is concatenation of all data in files,
# optionally filters out those rows that doesn't match chordRegex
def getTrainDataset(files, chordRegex=None, takeFunc = None):
    return getDataset10(files, chordRegex, takeFunc)

# returns list of groups - concatenation of all data grouped by song_id
def getTestDataset(files, regex=None, takeFunc=None):
    return getDataset10(files, regex, takeFunc).groupby(SONG_ID_COL)

# returns list of pairs (train, test)
# where train and tests are list of filenames
def getTrainTestPairs(files):
    l = len(files)
    res = []
    for i in range(0, l):
        mask = np.ones(l, dtype=bool)
        mask[i] = 0

        # take i-th group as a test set
        test = [files[i]]
        # take all groups except i-th as a train set
        train = list(itertools.compress(files, mask))

        res.append((train, test))
    return res

# cross validation by songs
def crossValBySongs(trainTestFiles, estimator, scorer):

    res = []
    i = 1

    for trainFiles, testFiles in trainTestFiles:
        print "Iteration: {}".format(i)

        print "Reading train dataset..."
        # join all songs in train set into single dataset
        train = getTrainDataset(trainFiles, filterChords5, None)

        X = train.ix[:, CHROMA_START_COL:]
        y = train.ix[:, CHORD_COL]

        # train
        print "Training..."
        estimator.fit(X, y)

        train = None
        X = None
        y = None
        gc.collect()

        # calculate score for each song
        # stores these scores in list
        scores = []
        print "Reading test dataset..."
        test = getTestDataset(testFiles, filterChordsNotNX)
        print "Testing..."
        scorer.reset()
        for song_id, df in test:
            X = df.ix[:, CHROMA_START_COL:]
            y = df.ix[:, CHORD_COL]
            timing = df.ix[:, TIME_COL]
            scorer.process_song(song_id, estimator, X, y, timing)

        test = None
        X = None
        y = None
        gc.collect()

        # take mean of all scores as a result
        score = scorer.score()
        res.append(score)
        i += 1

    return res

class AccuracyScorer:

    _scores = []
    _scorer = make_scorer(accuracy_score)

    def process_song(self, song_id, estimator, X, y, timing):
        score = self._scorer(estimator, X, y)
        print "song: {}; score={}".format(song_id, score)
        result.write(str(score)+',')
        self._scores.append(score)

    def score(self):
        mean = np.mean(self._scores)
        print "mean={}".format(mean)
        return mean

    def reset(self):
        self._scores = []

class ConfusionMatrixScorer:

    _y_true = []
    _y_pred = []
    _iter = 0

    def process_song(self, song_id, estimator, X, y, timing):
        self._y_true += list(y)
        self._y_pred += list(estimator.predict(X))

    def score(self):
        labels = np.unique(self._y_true + self._y_pred)
        matrix = confusion_matrix(self._y_true, self._y_pred, labels)
        df = pd.DataFrame(matrix, index=labels, columns=labels)
        self._iter += 1
        fn = "confusion_matrix{}.csv".format(self._iter)
        df.to_csv(fn)
        return matrix

    def reset(self):
        self._y_pred = []
        self._y_true = []

class SegmentCsrScorer():

    FRAME_LEN = 0.01

    _corr_time = 0.0
    _full_time = 0.0

    def _segment(self, chords, timing):
        segments = []
        i = 0
        ln = len(chords)
        while i < ln:
            j = i + 1
            while j < ln and chords[i] == chords[j]:
                j += 1
            chrd = chords[i]
            j_ = j if j != ln else j-1
            tm = float(timing[j_]) - float(timing[i])
            segments.append((chrd, tm))
            i = j
        return segments

    def _correct_timing(self, seg_true, seg_pred):
        len_true = len(seg_true)
        len_pred = len(seg_pred)

        seg_len = 0.0
        tm1 = 0; tm2 = 0
        i = 0; j = 0
        chrd1 = ''; chrd2 = ''
        while i < len_true and j < len_pred:
            if tm1 < self.FRAME_LEN:
                tm1 = seg_true[i][1]
                chrd1 = seg_true[i][0]
                i += 1
            if tm2 < self.FRAME_LEN:
                tm2 = seg_pred[j][1]
                chrd2 = seg_pred[j][0]
                j += 1

            tm = min(tm1, tm2)
            tm1 -= tm
            tm2 -= tm
            if chrd1 == chrd2:
                seg_len += tm

        return seg_len

    def process_song(self, song_id, estimator, X, y, timing):
        y_true = list(y)
        y_pred = list(estimator.predict(X))
        timing = list(timing)

        y_true_seg = self._segment(y_true, timing)
        y_pred_seg = self._segment(y_pred, timing)

        full_time = np.sum([seg[1] for seg in y_true_seg])
        corr_time = self._correct_timing(y_true_seg, y_pred_seg)
        self._full_time += full_time
        self._corr_time += corr_time

        print "song: {}; correct time: {:0.2f}; full time: {:0.2f}".format(song_id, corr_time, full_time)
        result.write( "song: {}; correct time: {:0.2f}; full time: {:0.2f}".format(song_id, corr_time, full_time) + '\n')
     #   result.write(str(score)+',')

    def score(self):
        csr = self._corr_time / self._full_time
        print "csr={:0.4f}".format(csr)
        return csr

    def reset(self):
        self._corr_time = 0.0
        self._full_time = 0.0

if __name__ == '__main__':

#    estimator = GaussianNB()
#    estimator = KNeighborsClassifier()
    estimator = SVC()
#    estimator = linear_model.Perceptron() 0.25
#    estimator = linear_model.SGDClassifier() 0.31
#    estimator = DecisionTreeClassifier() 0.27
#     scorer = SegmentCsrScorer()

#    scorer = ConfusionMatrixScorer()
    scorer = AccuracyScorer()
    groupsNum = 10
    files = ['collection/chroma{}.txt'.format(i) for i in range(1, 40)]
    trainTestFiles = getTrainTestPairs(files)
    # trainTestFiles = [(files[10:], files[:10])]

    scores = crossValBySongs(trainTestFiles, estimator, scorer)