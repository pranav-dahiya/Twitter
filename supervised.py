import pandas as pd
import numpy as np
from sklearn.svm import SVC, SVR
from sklearn.model_selection import cross_validate
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix
from sklearn.metrics import make_scorer


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
          # 'cf_matrix_02': make_scorer(cf_matrix_02),
          'cf_matrix_10': make_scorer(cf_matrix_10),
          'cf_matrix_11': make_scorer(cf_matrix_11),
          # 'cf_matrix_12': make_scorer(cf_matrix_12),
          # 'cf_matrix_20': make_scorer(cf_matrix_20),
          # 'cf_matrix_21': make_scorer(cf_matrix_21),
          # 'cf_matrix_22': make_scorer(cf_matrix_22),
          'accuracy': 'balanced_accuracy',
          'precision': 'precision_weighted',
          'recall': 'recall_weighted',
          'f1': 'f1_weighted'}

data = pd.read_json('data_formspring.json')

vectorizer = TfidfVectorizer(ngram_range=(2,3), stop_words='english')
X = vectorizer.fit_transform(data['content'])
y = data['label']
model = SVR()
model.fit(X, y)
data['ngram'] = pd.Series(model.predict(X))

# X = data[['profane_words', 'profanity_score', 'num_usertags',
#     'upper_case_density', 'sentiment', 'length', 'num_pronoun', 'ngram']]

X = data[['num_url', 'num_emoji', 'profane_words', 'profanity_score', 'num_exclamation_question',
    'num_stops', 'num_dash', 'num_star_dollar', 'num_ampersand', 'num_hashtags', 'num_usertags',
    'upper_case_density', 'sentiment', 'ngram']]

print(y.value_counts())

model = SVC(class_weight='balanced')
results = cross_validate(model, X, y, scoring=scores, cv=10, n_jobs=-1)
cf_matr = [[np.mean(results['test_cf_matrix_'+str(i)+str(j)]) for j in range(2)] for i in range(2)]
for row in cf_matr:
    for val in row:
        print(val, "&", end=" ")
    print()
print("${:.2f}".format(np.mean(results['test_accuracy'])), "\pm", "{:.2f}$".format(np.std(results['test_accuracy'])),
    "&", "${:.2f}".format(np.mean(results['test_precision'])), "\pm", "{:.2f}$".format(np.std(results['test_precision'])),
    "&", "${:.2f}".format(np.mean(results['test_f1'])), "\pm", "{:.2f}$".format(np.std(results['test_f1'])), "&",
    "${:.2f}".format(np.mean(results['test_recall'])), "\pm", "{:.2f}$".format(np.std(results['test_recall'])))

# class1 = data.loc[data['label'] == 0]
# class2 = data.loc[data['label'] == 1]
#
# for feature in ['num_url', 'num_emoji', 'profane_words', 'profanity_score', 'num_exclamation_question',
#     'num_stops', 'num_dash', 'num_star_dollar', 'num_ampersand', 'num_hashtags', 'num_usertags',
#     'upper_case_density', 'sentiment', ]
