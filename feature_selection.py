import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesClassifier

forest = ExtraTreesClassifier(n_estimators=250,
                              random_state=0)

data = pd.read_json('data.json')
feature_list = ['num_url', 'num_emoji', 'profane_words', 'profanity_score', 'num_exclamation_question',
    'num_stops', 'num_dash', 'num_star_dollar', 'num_ampersand', 'num_hashtags', 'num_usertags',
    'upper_case_density', 'sentiment', 'length', 'num_pronoun']
X = data[feature_list]
y = data['label']

forest.fit(X, y)
importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

print("Feature ranking:")

for f in range(X.shape[1]):
    print("%d. feature %s (%f)" % (f + 1, feature_list[indices[f]], importances[indices[f]]))


sorted_feature_list = [None for i in range(X.shape[1])]
for i, index in enumerate(indices):
    sorted_feature_list[i] = feature_list[index]

plt.figure()
plt.title("Feature importances")
plt.bar(range(X.shape[1]), importances[indices],
       color="r", yerr=std[indices], align="center")
plt.xticks(range(X.shape[1]), sorted_feature_list, rotation='vertical')
plt.xlim([-1, X.shape[1]])
plt.subplots_adjust(bottom=0.3)
plt.show()
