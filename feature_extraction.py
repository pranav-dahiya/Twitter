import warnings
warnings.filterwarnings("ignore")

import re
from nltk import word_tokenize, pos_tag
import pandas as pd
import numpy as np
from profanity_check import predict, predict_prob
from emot import emoji, emoticons
from bs4 import BeautifulSoup
from flair.models import TextClassifier, SequenceTagger
from flair.data import Sentence


def emoji_and_emoticon_count(string):
    num = 0
    for emotes in [emoji(string), emoticons(string)]:
        try:
            if emotes['flag']:
                num += len(emotes)
                for value, meaning in zip(emotes['value'], emotes['mean']):
                    string = string.replace(value, meaning)
        except:
            pass
    return {'content': string, 'num_emoji': num}


def num_urls(string):
    try:
        urls = re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\), ]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', string)
        for url in urls:
            string = string.replace(url, "url")
        return {'content': string, 'num_url': len(urls)}
    except:
        print(string)
        exit(0)


def num_profane_words(string):
    words = string.split()
    profanity = predict(words)
    return np.sum(profanity)


def upper_case_density(string):
    total = 0
    upper = 0
    for char in string:
        if char.islower():
            total += 1
        if char.isupper():
            total += 1
            upper += 1
    return upper/(total + 10**(-6))


def num_pronouns(string):
    num = 0
    for word, tag in pos_tag(word_tokenize(string)):
        if tag == 'PRP':
            num += 1
    return num


# data = pd.read_json("Data/dataset-for-detection-of-cybertrolls/Dataset for Detection of Cyber-Trolls.json", lines=True)
#
# data['label'] = data['annotation'].map(lambda row: row['label'][0])
# data = data[['content', 'label']]

# data = pd.read_csv("Data/Twitter/labeled_data.csv")
# data['content'] = data['tweet']
# data['label'] = data['class']
# print(data.head)
# data = data[['content', 'label']]

data = pd.read_csv("Data/formspring-data-for-cyberbullying-detection/formspring_data.csv", delimiter="\t")
print(data.columns)
data['content'] = data['ques']
data['label'] = data.apply(lambda row: int(row['ans1'] == 'Yes' or row['ans2'] == 'Yes' or row['ans3'] == 'Yes'), axis=1)
print(data.head)
data = data[['content', 'label']]
data = data.dropna()
print(data['label'].value_counts())

url_df = pd.DataFrame(data['content'].map(num_urls).to_list())
data['content'] = url_df['content']
data['num_url'] = url_df['num_url']

data = data[data['content'].apply(lambda string: type(string) == type(""))]

data['content'] = data['content'].map(lambda string: BeautifulSoup(string, features="html.parser").text)

data = data[data['content'].apply(lambda string: type(string) == type(""))]

emoji_df = pd.DataFrame(data['content'].map(emoji_and_emoticon_count).to_list())
data['content'] = emoji_df['content']
data['num_emoji'] = emoji_df['num_emoji']

data = data[data['content'].apply(lambda string: type(string) == type(""))]

data['profane_words'] = data['content'].map(num_profane_words)

data['profanity_score'] = predict_prob(data['content'].to_numpy())

data['num_exclamation_question'] = data['content'].map(lambda string: string.count("!") + string.count("?"))

data['num_stops'] = data['content'].map(lambda string: string.count("."))

data['num_dash'] = data['content'].map(lambda string: string.count("-"))

data['num_star_dollar'] = data['content'].map(lambda string: string.count("*") + string.count("$"))

data['num_ampersand'] = data['content'].map(lambda string: string.count("&"))

data['num_hashtags'] = data['content'].map(lambda string: string.count("#"))

data['num_usertags'] = data['content'].map(lambda string: string.count("@"))

data['upper_case_density'] = data['content'].map(upper_case_density)

flair_sentiment = TextClassifier.load('en-sentiment')
sentences = data['content'].map(lambda string: Sentence(string)).to_list()
flair_sentiment.predict(sentences)
sign = {'POSITIVE': 1, 'NEGATIVE': -1}
sentiments = [sign[sentence.labels[0].value]*sentence.labels[0].score for sentence in sentences]
data['sentiment'] = pd.Series(sentiments)

data['length'] = data['content'].map(lambda string: len(string))

data['num_pronoun'] = data['content'].map(num_pronouns)

data.to_json('data_formspring.json')

print(data.head)
