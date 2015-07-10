"""
    microarray.py - validação cruzada da classificação de dados de microarray

    Pedro Sousa Lacerda (UFABC, 2015)
    pslacerda@gmail.com



    DISCUSSÃO DO CÓDIGO

    O código que está lendo é baseado neste outro:

            http://github.com/jrf/microarray

            (avise-me se souber quem é o autor)

    A. Selecionamos os genes mais "importantes". Para isto fazemos uma
       classificação prévia com árvores aleatóreas de decisão treinadas com a
       matriz de expressão gênica,

    B. Fazemos a validação cruzada de um conjunto de classificadores (SVC, NB,
       KNN, ...) considerando apenas os genes selecionados.

    C. Consideramos primeiro apenas o gene mais importante, depois os dois
       mais importantes ... até os NFEATURES mais importantes.


    O resultado é uma figura comparando os vários métodos de classificação na
    predição hipotética de doenças utilizando (1) microarray,expressão gênica; e
    (2) aprendizagem de máquina, mineração de dados.



    INSTRUÇÕES DE USO
        python3 microarray.py NFOLDS NFEATURES SAMPLES_CSV LABELS_CSV OUTPUT_IMG


    EXEMPLO DE USO
        python3 microarray.py 10 20 khan2001x.csv khan2001y.csv out.pdf


    DEFEITOS CONHECIDOS
      Talvez a variável num_features possa desmembrada em duas outras? Ignore
    com cuidado se encontrar este WARNING:, corrija sua linha de comandos.

        The least populated class in y has only 8 members, which is too few. The
        minimum number of labels for any class cannot be less than n_folds=10.

    Para o programa "ficar completo" faltaria salvar os modelos num arquivo para
    poder reusarmos posteriormente com o intuito de classificar novos pacientes,
    auxiliando no diagnóstico de doenças. (e retroalimentá-lo quando confirmada
    ou negada a doença).


    LINKS RELACIONADOS
      Outro protocolo (talvez melhor que este?):
        http://bioinformatics.oxfordjournals.org/content/21/19/3755.full.pdf

      Datasets:
        http://github.com/ramhiser/datamicroarray/wiki
        http://www.dcs.warwick.ac.uk/~yina/Datasets.htm

"""

import os
import sys

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

if (len(sys.argv) != 6):
    print("USAGE: %s NFOLDS NFEATURES X_CSV Y_CSV OUTPUT_IMAGE" % sys.argv[0])
    print("EXAMPLE: %s 10 20 khan2001x.csv khan2001y.csv out.pdf" % sys.argv[0])
    print()
    sys.exit(0)

NFOLDS = int(sys.argv[1])
assert NFOLDS > 0

NFEATURES = int(sys.argv[2])
assert NFEATURES > 0

SAMPLES_CSV = sys.argv[3]
assert os.access(SAMPLES_CSV, os.R_OK)

LABELS_CSV = sys.argv[4]
assert os.access(LABELS_CSV, os.R_OK)

OUTPUT_IMG = sys.argv[5]
# assert os.access(OUTPUT_IMAGE, os.W_OK)





# Read in expression matrix
Xkhan = pd.read_csv(SAMPLES_CSV, index_col=0)
Xkhan = Xkhan.values

# Read in sample categories
Ykhan = pd.read_csv(LABELS_CSV, index_col=0)
Ykhan = pd.Categorical.from_array(Ykhan.values[:,0])
Ykhan = Ykhan.codes




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


import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


num_features = NFEATURES
clfs = [lambda:SVC(kernel='linear'), GaussianNB]
# clfs = [lambda:SVC(kernel='linear'), GaussianNB, NearestCentroid,
#          KNeighborsClassifier, LDA, RandomForestClassifier]

label = ['SVC', 'NB', 'NC', 'KNN', 'LDA', 'RF']
color = ['#348ABD', '#A60628', '#7A68A6', '#467821', '#D55E00', '#CC79A7']

accS = np.zeros((5, num_features + 1))
for i, clf in enumerate(clfs):
    for j in range(1, num_features+1):
        print('classifier=', label[i], ', nfeatures=ngenes=', j, sep='')

        scores = []
        skf = cross_validation.StratifiedKFold(Ykhan, n_folds=NFOLDS)
        for train, test in skf:
            X_train, X_test = Xkhan[train], Xkhan[test]
            y_train, y_test = Ykhan[train], Ykhan[test]
            XRF_train, imp, ind, std = fitRF(X_train, y_train, est=2000)  # RFsel
            XRF_test = X_test[:, ind]  # reorder test set after RFsel
            clf2 = clfs[i]()
            clf2.fit(XRF_train[:, 0:j], y_train)
            scores.append(clf2.score(XRF_test[:, 0:j], y_test))
        score = np.mean(scores)
        accS[i, j] = score

    plt.plot(np.arange(0, num_features + 1), accS[i, :], color[i],
             label=label[i], alpha=0.85, lw=2)

# accS = accS[:, 1:num_features + 1]  # chop the leading zeros


plt.xlim([1, num_features])
plt.xticks(np.arange(1, num_features + 1), size='small')
plt.ylabel('Accuracy', fontsize=8)
plt.xlabel('Number of Genes (SRBCT)', fontsize=8)
plt.tick_params(axis='both', which='major', labelsize=8)
plt.tick_params(axis='both', which='minor', labelsize=8)
plt.subplots_adjust(hspace=.5)
plt.legend(numpoints=1, ncol=6, loc=4, fontsize=8)


plt.savefig(OUTPUT_IMG)

