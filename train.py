# Monika Kisz - training

from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn import preprocessing, metrics
from random_forest import RandomForest
import sklearn.utils as sku

df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/nursery/nursery.data",
                 names=["parents", "has_nurs", "form", "children", "housing", "finance", "social", "health", "class"])
enc = preprocessing.OrdinalEncoder(dtype=np.int64, categories=[['usual', 'pretentious', 'great_pret'],
                                                               ['proper', 'less_proper', 'improper', 'critical',
                                                                'very_crit'],
                                                               ['complete', 'completed', 'incomplete', 'foster'],
                                                               ['1', '2', '3', 'more'],
                                                               ['convenient', 'less_conv', 'critical'],
                                                               ['convenient', 'inconv'],
                                                               ['nonprob', 'slightly_prob', 'problematic'],
                                                               ['recommended', 'priority', 'not_recom']])
data = enc.fit_transform(df[['parents', 'has_nurs', 'form', 'children', 'housing', 'finance', 'social', 'health']])

enc2 = preprocessing.OrdinalEncoder(dtype=np.int64,
                                    categories=[['recommend', 'priority', 'not_recom', 'very_recom', 'spec_prior']])
target = enc2.fit_transform(df[['class']])

X, y = data, target
X, y = sku.shuffle(X, y)
y = y.flatten()
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.5, random_state=2
)
clf = RandomForest()
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)


# function to check accuracy for every probe, without dividing per class
def accuracy(y_test, y_pred):
    return np.sum(y_test == y_pred) / len(y_test)


acc = accuracy(y_test, predictions)
print(acc)

cnf_matrix = metrics.confusion_matrix(y_test, predictions)
print(cnf_matrix)

FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)
FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
TP = np.diag(cnf_matrix)
TN = cnf_matrix.sum() - (FP + FN + TP)
FP = FP.astype(float)
FN = FN.astype(float)
TP = TP.astype(float)
TN = TN.astype(float)

# Sensitivity, hit rate, recall, or true positive rate
TPR = np.divide(TP, (TP + FN), out=np.zeros_like(TP), where=(TP + FN) != 0)
print(f'TPR {TPR}')

# Precision or positive predictive value
PPV = np.divide(TP, (TP + FP), out=np.zeros_like(TP), where=(TP + FP) != 0)
print(f'PPV {PPV}')

# Fall out or false positive rate
FPR = np.divide(FP, (FP + TN), out=np.zeros_like(FP), where=(FP + TN) != 0)
print(f'FPR {FPR}')

# Overall accuracy for each class
ACC = np.divide((TP + TN), (TP + FP + FN + TN), out=np.zeros_like(TP + TN), where=(TP + FP + FN + TN) != 0)
print(f'ACC {ACC}')

# F1 score
F1 = np.divide((2 * PPV * TPR), (PPV + TPR), out=np.zeros_like(2 * PPV * TPR), where=(PPV + TPR) != 0)
print(f'F1 {F1}')
