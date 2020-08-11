import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_validate


class LemmaTokenizer:
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]


data = pd.read_json('data.json')

X = data['content']
y = data['label']

print(np.sum(y)/len(y))

vectorizer = TfidfVectorizer()
X_ = vectorizer.fit_transform(X)
model = LogisticRegression()
results = cross_validate(model, X_, y, scoring=('accuracy', 'precision', 'recall',
    'f1'), cv=10, n_jobs=-1)
print("${:.2f}".format(np.mean(results['test_accuracy'])), "\pm", "{:.2f}$".format(np.std(results['test_accuracy'])),
    "&", "${:.2f}".format(np.mean(results['test_precision'])), "\pm", "{:.2f}$".format(np.std(results['test_precision'])),
    "&", "${:.2f}".format(np.mean(results['test_f1'])), "\pm", "{:.2f}$".format(np.std(results['test_f1'])), "&",
    "${:.2f}".format(np.mean(results['test_recall'])), "\pm", "{:.2f}$".format(np.std(results['test_recall'])))

vectorizer = TfidfVectorizer(stop_words='english')
X_ = vectorizer.fit_transform(X)
model = LogisticRegression()
results = cross_validate(model, X_, y, scoring=('accuracy', 'precision', 'recall',
    'f1'), cv=10, n_jobs=-1)
print("${:.2f}".format(np.mean(results['test_accuracy'])), "\pm", "{:.2f}$".format(np.std(results['test_accuracy'])),
    "&", "${:.2f}".format(np.mean(results['test_precision'])), "\pm", "{:.2f}$".format(np.std(results['test_precision'])),
    "&", "${:.2f}".format(np.mean(results['test_f1'])), "\pm", "{:.2f}$".format(np.std(results['test_f1'])), "&",
    "${:.2f}".format(np.mean(results['test_recall'])), "\pm", "{:.2f}$".format(np.std(results['test_recall'])))

vectorizer = TfidfVectorizer(tokenizer=LemmaTokenizer(), stop_words='english')
X_ = vectorizer.fit_transform(X)
model = LogisticRegression()
results = cross_validate(model, X_, y, scoring=('accuracy', 'precision', 'recall',
    'f1'), cv=10, n_jobs=-1)
print("${:.2f}".format(np.mean(results['test_accuracy'])), "\pm", "{:.2f}$".format(np.std(results['test_accuracy'])),
    "&", "${:.2f}".format(np.mean(results['test_precision'])), "\pm", "{:.2f}$".format(np.std(results['test_precision'])),
    "&", "${:.2f}".format(np.mean(results['test_f1'])), "\pm", "{:.2f}$".format(np.std(results['test_f1'])), "&",
    "${:.2f}".format(np.mean(results['test_recall'])), "\pm", "{:.2f}$".format(np.std(results['test_recall'])))

for i in range(1,5):
    for j in range(i,5):
        vectorizer = TfidfVectorizer(ngram_range=(i,j), stop_words='english')
        X_ = vectorizer.fit_transform(X)
        model = LogisticRegression()
        results = cross_validate(model, X_, y, scoring=('accuracy', 'precision', 'recall',
            'f1'), cv=10, n_jobs=-1)
        print(i, "&", j, "&", "${:.2f}".format(np.mean(results['test_accuracy'])), "\pm", "{:.2f}$".format(np.std(results['test_accuracy'])),
            "&", "${:.2f}".format(np.mean(results['test_precision'])), "\pm", "{:.2f}$".format(np.std(results['test_precision'])),
            "&", "${:.2f}".format(np.mean(results['test_f1'])), "\pm", "{:.2f}$".format(np.std(results['test_f1'])), "&",
            "${:.2f}".format(np.mean(results['test_recall'])), "\pm", "{:.2f}$".format(np.std(results['test_recall'])))

vectorizer = TfidfVectorizer(ngram_range=(2,3), stop_words='english')
X_ = vectorizer.fit_transform(X)
models = [LogisticRegression(), KNeighborsClassifier(), SVC(kernel='linear'),
    SVC(kernel='poly'), SVC(), DecisionTreeClassifier()]
for model in models:
    results = cross_validate(model, X_, y, scoring=('accuracy', 'precision', 'recall',
        'f1'), cv=10, n_jobs=-1)
    print("${:.2f}".format(np.mean(results['test_accuracy'])), "\pm", "{:.2f}$".format(np.std(results['test_accuracy'])),
        "&", "${:.2f}".format(np.mean(results['test_precision'])), "\pm", "{:.2f}$".format(np.std(results['test_precision'])),
        "&", "${:.2f}".format(np.mean(results['test_f1'])), "\pm", "{:.2f}$".format(np.std(results['test_f1'])), "&",
        "${:.2f}".format(np.mean(results['test_recall'])), "\pm", "{:.2f}$".format(np.std(results['test_recall'])))
