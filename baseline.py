import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import cross_validate
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix
from sklearn.metrics import make_scorer
import numpy as np

def cf_matrix_00(y_true, y_pred):
    cf_matr = confusion_matrix(y_true, y_pred)
    return cf_matr[0,0] / np.sum(cf_matr[0,:])
def cf_matrix_01(y_true, y_pred):
    cf_matr = confusion_matrix(y_true, y_pred)
    return cf_matr[0,1] / np.sum(cf_matr[0,:])
def cf_matrix_02(y_true, y_pred):
    cf_matr = confusion_matrix(y_true, y_pred)
    return cf_matr[0,2] / np.sum(cf_matr[0,:])
def cf_matrix_10(y_true, y_pred):
    cf_matr = confusion_matrix(y_true, y_pred)
    return cf_matr[1,0] / np.sum(cf_matr[1,:])
def cf_matrix_11(y_true, y_pred):
    cf_matr = confusion_matrix(y_true, y_pred)
    return cf_matr[1,1] / np.sum(cf_matr[1,:])
def cf_matrix_12(y_true, y_pred):
    cf_matr = confusion_matrix(y_true, y_pred)
    return cf_matr[1,2] / np.sum(cf_matr[1,:])
def cf_matrix_20(y_true, y_pred):
    cf_matr = confusion_matrix(y_true, y_pred)
    return cf_matr[2,0] / np.sum(cf_matr[2,:])
def cf_matrix_21(y_true, y_pred):
    cf_matr = confusion_matrix(y_true, y_pred)
    return cf_matr[2,1] / np.sum(cf_matr[2,:])
def cf_matrix_22(y_true, y_pred):
    cf_matr = confusion_matrix(y_true, y_pred)
    return cf_matr[2,2] / np.sum(cf_matr[2,:])


scores = {'cf_matrix_00': make_scorer(cf_matrix_00),
          'cf_matrix_01': make_scorer(cf_matrix_01),
          'cf_matrix_02': make_scorer(cf_matrix_02),
          'cf_matrix_10': make_scorer(cf_matrix_10),
          'cf_matrix_11': make_scorer(cf_matrix_11),
          'cf_matrix_12': make_scorer(cf_matrix_12),
          'cf_matrix_20': make_scorer(cf_matrix_20),
          'cf_matrix_21': make_scorer(cf_matrix_21),
          'cf_matrix_22': make_scorer(cf_matrix_22),
          'accuracy': 'balanced_accuracy',
          'precision': 'precision_weighted',
          'recall': 'recall_weighted',
          'f1': 'f1_weighted'}

# data = pd.read_json("Data/dataset-for-detection-of-cybertrolls/Dataset for Detection of Cyber-Trolls.json", lines=True)
#
# data['label'] = data['annotation'].map(lambda row: row['label'][0])
# data = data[['content', 'label']]

data = pd.read_csv("Data/Twitter/labeled_data.csv")
data['content'] = data['tweet']
data['label'] = data['class']
data = data[['content', 'label']]

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data['content'])
y = data['label']
model = SVC()

results = cross_validate(model, X, y, scoring=scores, cv=10, n_jobs=-1)
cf_matr = [[np.mean(results['test_cf_matrix_'+str(i)+str(j)]) for j in range(3)] for i in range(3)]
for row in cf_matr:
    for val in row:
        print("${:.2f}".format(val), "&", end=" ")
    print()
print("${:.2f}".format(np.mean(results['test_accuracy'])), "\pm", "{:.2f}$".format(np.std(results['test_accuracy'])),
    "&", "${:.2f}".format(np.mean(results['test_precision'])), "\pm", "{:.2f}$".format(np.std(results['test_precision'])),
    "&", "${:.2f}".format(np.mean(results['test_f1'])), "\pm", "{:.2f}$".format(np.std(results['test_f1'])), "&",
    "${:.2f}".format(np.mean(results['test_recall'])), "\pm", "{:.2f}$".format(np.std(results['test_recall'])))
