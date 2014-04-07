import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# Read in Singh data
xsingh = pd.read_csv("singh2002x.csv", index_col=0)
ysingh = pd.read_csv("singh2002y.csv", index_col=0)
ysingh = ysingh.rename(columns={'x': 'y'})

# Convert to numpy arrays
Xsingh = np.array(xsingh)
ysingh = np.array(ysingh)

# Create an array of values, where 0 corresponds to 'Tumor' and 1 corresponds
# to 'Normal'

Ysingh = []
for i in np.arange(0, len(ysingh)):
    if ysingh[i] == 'Tumor':
        Ysingh.append(0)
    elif ysingh[i] == 'Normal':
        Ysingh.append(1)
Ysingh = np.array(Ysingh)


# Read in Khan data
xkhan = pd.read_csv("khan2001x.csv", index_col=0)
ykhan = pd.read_csv("khan2001y.csv", index_col=0)
ykhan = ykhan.rename(columns={'x': 'y'})

# Convert to numpy arrays
Xkhan = np.array(xkhan)
ykhan = np.array(ykhan)

# Create an array of values, where 0 corresponds to 'EWS', 1 corresponds to
# 'RMS' 2 corresponds to 'NB' and 3 corresponds to 'BL'

Ykhan = []
for i in np.arange(0, len(ykhan)):
    if ykhan[i] == 'EWS':
        Ykhan.append(0)
    elif ykhan[i] == 'RMS':
        Ykhan.append(1)
    elif ykhan[i] == 'NB':
        Ykhan.append(2)
    elif ykhan[i] == 'BL':
        Ykhan.append(3)
Ykhan = np.array(Ykhan)


# In[4]:

# Read in Golub data
xgolub = pd.read_csv("golub1999x.csv", index_col=0)
ygolub = pd.read_csv("golub1999y.csv", index_col=0)
ygolub = ygolub.rename(columns={'x': 'y'})

# Convert to numpy arrays
Xgolub = np.array(xgolub)
ygolub = np.array(ygolub)

# Create an array of values, where 0 corresponds to 'ALL' and 1 corresponds to
# 'AML'

Ygolub = []
for i in np.arange(0, len(ygolub)):
    if ygolub[i] == 'ALL':
        Ygolub.append(0)
    elif ygolub[i] == 'AML':
        Ygolub.append(1)
Ygolub = np.array(Ygolub)


# In[5]:

from sklearn.ensemble import RandomForestClassifier

# fitRF returns: (1) the microarray ordered by feature importances (2) the
# feature importances themselves (3) the genes ranked by feature importances
# (4) the standard errors of the feature importances


def fitRF(X, Y, est=2000):
    forest = RandomForestClassifier(n_estimators=est,
                                    max_features='sqrt',
                                    min_samples_split=1,
                                    oob_score=True,
                                    n_jobs=-1,
                                    random_state=0)
    forest.fit(X, Y)  # fit the random forest
    importances = forest.feature_importances_  # feature importances
    std = np.std([tree.feature_importances_ for tree in forest.estimators_],
                 axis=0)  # standard errors of feature importances
    # sort the importances by index, then reverse the array to obtain feature
    # importances ordered from largest to smallest
    indices = np.argsort(importances)[::-1]

    X = X[:, indices]  # reorder the microarray data by the ordered indices
    return(X, importances, indices, std)


from sklearn import cross_validation
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.neighbors import KNeighborsClassifier
from sklearn.lda import LDA


def SVC_select_cv(X, Y, num_features):
    scores = []
    # kf = cross_validation.KFold(Y.size, n_folds=10)
    skf = cross_validation.StratifiedKFold(Y, n_folds=10)
    for train, test in skf:
        X_train, X_test, y_train, y_test = X[train], X[test], Y[train], Y[test]
        XRF_train, imp, ind, std = fitRF(X_train, y_train, est=2000)  # RFsel
        XRF_test = X_test[:, ind]  # reorder test set after RFsel
        clf = SVC(kernel='linear')
        clf.fit(XRF_train[:, 0:num_features], y_train)
        scores.append(clf.score(XRF_test[:, 0:num_features], y_test))
    score = np.mean(scores)
    return(score)


def GNB_select_cv(X, Y, num_features):
    scores = []
    skf = cross_validation.StratifiedKFold(Y, n_folds=10)
    for train, test in skf:
        X_train, X_test, y_train, y_test = X[train], X[test], Y[train], Y[test]
        XRF_train, imp, ind, std = fitRF(X_train, y_train, est=2000)  # RFsel
        XRF_test = X_test[:, ind]  # reorder test set after RFsel
        clf = GaussianNB()
        clf.fit(XRF_train[:, 0:num_features], y_train)
        scores.append(clf.score(XRF_test[:, 0:num_features], y_test))
    score = np.mean(scores)
    return(score)


def NC_select_cv(X, Y, num_features):
    scores = []
    skf = cross_validation.StratifiedKFold(Y, n_folds=10)
    for train, test in skf:
        X_train, X_test, y_train, y_test = X[train], X[test], Y[train], Y[test]
        XRF_train, imp, ind, std = fitRF(X_train, y_train, est=2000)  # RFsel
        XRF_test = X_test[:, ind]  # reorder test set after RFsel
        clf = NearestCentroid()
        clf.fit(XRF_train[:, 0:num_features], y_train)
        scores.append(clf.score(XRF_test[:, 0:num_features], y_test))
    score = np.mean(scores)
    return(score)


def KNN_select_cv(X, Y, num_features):
    scores = []
    skf = cross_validation.StratifiedKFold(Y, n_folds=10)
    for train, test in skf:
        X_train, X_test, y_train, y_test = X[train], X[test], Y[train], Y[test]
        XRF_train, imp, ind, std = fitRF(X_train, y_train, est=2000)  # RFsel
        XRF_test = X_test[:, ind]  # reorder test set after RFsel
        clf = KNeighborsClassifier()
        clf.fit(XRF_train[:, 0:num_features], y_train)
        scores.append(clf.score(XRF_test[:, 0:num_features], y_test))
    score = np.mean(scores)
    return(score)


def LDA_select_cv(X, Y, num_features):
    scores = []
    skf = cross_validation.StratifiedKFold(Y, n_folds=10)
    for train, test in skf:
        X_train, X_test, y_train, y_test = X[train], X[test], Y[train], Y[test]
        XRF_train, imp, ind, std = fitRF(X_train, y_train, est=2000)  # RFsel
        XRF_test = X_test[:, ind]  # reorder test set after RFsel
        clf = LDA()
        clf.fit(XRF_train[:, 0:num_features], y_train)
        scores.append(clf.score(XRF_test[:, 0:num_features], y_test))
    score = np.mean(scores)
    return(score)


import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

num_features = 20

print("\nGolub data\n")
accL = np.zeros((5, num_features + 1))
for j in np.arange(1, num_features + 1):
    print("SVC classifier with num_features:", j)
    accL[0, j] = SVC_select_cv(Xgolub, Ygolub, num_features=j)
for j in np.arange(1, num_features + 1):
    print("GaussianNB classifier with num_features:", j)
    accL[1, j] = GNB_select_cv(Xgolub, Ygolub, num_features=j)
for j in np.arange(1, num_features + 1):
    print("NearestCentroid classifier with num_features:", j)
    accL[2, j] = NC_select_cv(Xgolub, Ygolub, num_features=j)
for j in np.arange(1, num_features + 1):
    print("KNearestNeighbors classifier with num_features:", j)
    accL[3, j] = KNN_select_cv(Xgolub, Ygolub, num_features=j)
for j in np.arange(1, num_features + 1):
    print("LDA classifier with num_features:", j)
    accL[4, j] = LDA_select_cv(Xgolub, Ygolub, num_features=j)
accL = accL[:, 1:num_features + 1]  # chop the leading zeros

print("\nKhan data\n")
accS = np.zeros((5, num_features + 1))
for j in np.arange(1, num_features + 1):
    print("SVC classifier with num_features:", j)
    accS[0, j] = SVC_select_cv(Xkhan, Ykhan, num_features=j)
for j in np.arange(1, num_features + 1):
    print("GaussianNB classifier with num_features:", j)
    accS[1, j] = GNB_select_cv(Xkhan, Ykhan, num_features=j)
for j in np.arange(1, num_features + 1):
    print("NearestCentroid classifier with num_features:", j)
    accS[2, j] = NC_select_cv(Xkhan, Ykhan, num_features=j)
for j in np.arange(1, num_features + 1):
    print("KNearestNeighbors classifier with num_features:", j)
    accS[3, j] = KNN_select_cv(Xkhan, Ykhan, num_features=j)
for j in np.arange(1, num_features + 1):
    print("LDA classifier with num_features:", j)
    accS[4, j] = LDA_select_cv(Xkhan, Ykhan, num_features=j)
accS = accS[:, 1:num_features + 1]  # chop the leading zeros

print("\nSingh data\n")
accP = np.zeros((5, num_features + 1))
for j in np.arange(1, num_features + 1):
    print("SVC classifier with num_features:", j)
    accP[0, j] = SVC_select_cv(Xsingh, Ysingh, num_features=j)
for j in np.arange(1, num_features + 1):
    print("GaussianNB classifier with num_features:", j)
    accP[1, j] = GNB_select_cv(Xsingh, Ysingh, num_features=j)
for j in np.arange(1, num_features + 1):
    print("NearestCentroid classifier with num_features:", j)
    accP[2, j] = NC_select_cv(Xsingh, Ysingh, num_features=j)
for j in np.arange(1, num_features + 1):
    print("KNearestNeighbors classifier with num_features:", j)
    accP[3, j] = KNN_select_cv(Xsingh, Ysingh, num_features=j)
for j in np.arange(1, num_features + 1):
    print("LDA classifier with num_features:", j)
    accP[4, j] = LDA_select_cv(Xsingh, Ysingh, num_features=j)
accP = accP[:, 1:num_features + 1]  # chop the leading zeros


color = ['#348ABD', '#A60628', '#7A68A6', '#467821', '#D55E00', '#CC79A7']
label = ['SVC', 'NB', 'NC', 'KNN', 'LDA', 'RF']

# Leukemia
plt.figure(figsize=(6, 9))
plt.figure
plt.subplot(311)
for i in np.arange(0, 5):
    plt.plot(np.arange(1, num_features + 1), accL[i, :], color[i],
             label=label[i], alpha=0.85, lw=2)
plt.xlim([1, num_features])
plt.xticks(np.arange(1, num_features + 1), size='small')
plt.ylabel('Accuracy', fontsize=8)
plt.xlabel('Number of Genes (Leukemia)', fontsize=8)

plt.suptitle('Stratified 10-Fold Cross-Validation Accuracy of Various\
             Classifiers', fontsize=12)

plt.tick_params(axis='both', which='major', labelsize=8)
plt.tick_params(axis='both', which='minor', labelsize=8)
plt.subplots_adjust(hspace=.5)
# plt.tick_params(\
#    axis='x',          # changes apply to the x-axis
#    which='both',      # both major and minor ticks are affected
#    bottom='off',      # ticks along the bottom edge are off
#    top='off',         # ticks along the top edge are off
#   labelbottom='off') # labels along the bottom edge are off
plt.legend(numpoints=1, ncol=6, loc=4, fontsize=8)
# plt.legend(bbox_to_anchor=(1.3, 1))

# SRBCT
plt.subplot(312)
for i in np.arange(0, 5):
    plt.plot(np.arange(1, num_features + 1), accS[i, :], color[i],
             label=label[i], alpha=0.85, lw=2)
plt.xlim([1, num_features])
plt.xticks(np.arange(1, num_features + 1), size='small')
plt.ylabel('Accuracy', fontsize=8)
plt.xlabel('Number of Genes (SRBCT)', fontsize=8)
plt.tick_params(axis='both', which='major', labelsize=8)
plt.tick_params(axis='both', which='minor', labelsize=8)
plt.subplots_adjust(hspace=.5)
# plt.tick_params(\
#    axis='x',          # changes apply to the x-axis
#    which='both',      # both major and minor ticks are affected
#    bottom='off',      # ticks along the bottom edge are off
#    top='off',         # ticks along the top edge are off
#    labelbottom='off') # labels along the bottom edge are off
plt.legend(numpoints=1, ncol=6, loc=4, fontsize=8)

# Prostate
plt.subplot(313)
for i in np.arange(0, 5):
    plt.plot(np.arange(1, num_features + 1), accP[i, :], color[i],
             label=label[i], alpha=0.85, lw=2)
plt.xlim([1, num_features])
plt.xticks(np.arange(1, num_features + 1), size='small')
plt.ylabel('Accuracy', fontsize=8)
plt.xlabel('Number of Genes (Prostate)', fontsize=8)
plt.legend(numpoints=1, ncol=6, loc=4, fontsize=8)
plt.tick_params(axis='both', which='major', labelsize=8)
plt.tick_params(axis='both', which='minor', labelsize=8)
# plt.tick_params(\
#    axis='x',          # changes apply to the x-axis
#    which='both',      # both major and minor ticks are affected
#    bottom='off',      # ticks along the bottom edge are off
#    top='off',         # ticks along the top edge are off
#    labelbottom='off') # labels along the bottom edge are off
# plt.legend(bbox_to_anchor=(1.3, 1))
# plt.show()
plt.savefig('cv.png')
plt.savefig('cv.pdf')


# CSS styling

from IPython.core.display import HTML


def css_styling():
    styles = open("./custom.css", "r").read()
    return HTML(styles)
css_styling()
