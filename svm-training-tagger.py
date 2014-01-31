# -*- coding: utf-8 -*-
import json
import pickle
import re

from sklearn import svm
from sklearn import cross_validation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.externals import joblib
from numpy import array

# 1) Reading & Parsing news.json
# 2) Taking only those news: with enough body, for top-X categories until Max amount of them
# 3) Train & Tests
# 4) Adding more categories

"""
Corpus size: 29539
Corpus matrix shape: (29539, 125292)
Labels vector shape: (29539L,)
Supported vectors length: (12219, 125292)
Dual coef. length: (9, 12219)
Score k-fold validation: 0.7717%
(1L, 125292L)
[u'/economia']
"""

def insert_processed_review(corpus_labels, json_news, stats):
    """
        TODO Filtering misencoded/short/foreigner ones.
    """
    news = json.loads(json_news)
    category = news['category']
    if category in stats:
        # http://www.tamasoft.co.jp/en/general-info/unicode.html
        text = news['title'] + " " + news['body']
        text = re.sub('\x93', ' ', text) # “
        text = re.sub('\x94', ' ', text) # ”
        text = re.sub('\x91', ' ', text) # ‘
        text = re.sub('\x92', ' ', text) # ’
        text = re.sub('\'', ' ', text) # ’
        text = re.sub('\xe1', 'a', text) # ’
        text = re.sub('\xe9', 'e', text) # ’
        text = re.sub('\xed', 'i', text) # ’
        text = re.sub('\xf3', 'o', text) # ’
        text = re.sub('\xfa', 'u', text) # ’
        text = re.sub('\xc1', 'A', text) # ’
        text = re.sub('\xc9', 'E', text) # ’
        text = re.sub('\xcd', 'I', text) # ’
        text = re.sub('\xd3', 'O', text) # ’
        text = re.sub('\xda', 'U', text) # ’
        text = re.sub('\xfc', 'u', text) # ’
        text = re.sub('\xdc', 'U', text) # ’
        text = re.sub('\xf1', u'ñ', text) # ’
        text = re.sub('\xd1', u'Ñ', text) # ’
        text = re.sub('\xe7', u'ç', text) # ’
        text = re.sub('\xc7', u'Ç', text) # ’

        words = re.findall(r'\b[a-zA-Z]+\b', text) #_ActionsContainer \b(\w+)\b
        words = [w for w in words if len(w) > 1]
        if len(words) > 20 and stats[category] < 6000:
            corpus_labels[' '.join(words)] = category
            stats[category] += 1

corpus_labels = {} # {"Rajoy confirma una subida...": "/politica"}
stats = {"/espana": 0, "/politica": 0, "/economia": 0, "/sociedad": 0, "/mercados-cotizaciones": 0, "/futbol": 0,
         "/internacional": 0, "/legislacion": 0, "/empresas-finanzas": 0, "/salud": 0}

with open("data/tagger/news.json", "r") as file_in:
    for line in file_in.readlines():  # 1 review/dict per line!
        insert_processed_review(corpus_labels, line, stats)

print("Corpus size: %d " % len(corpus_labels))

# Vectorization: transforming text corpora to TF matrices
# http://scikit-learn.org/stable/modules/feature_extraction.html#common-vectorizer-usage
vectorizer = CountVectorizer(min_df=1)
X = vectorizer.fit_transform(corpus_labels.keys())  # vocabulary generated for this input corpus
#X.toarray()

# Transform list of labels to an array
y = array(corpus_labels.values())

print("Corpus matrix shape: %s " % str(X.shape))
print("Labels vector shape: %s " % str(y.shape))

# Training and validation data (k-fold = 30%)
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.3, random_state=23)


# With a RBF kernel, gamma=0.01, corpus_number_perdomain = 5000 and
# vectors_size = 20 to reach the maximum score, however we
# use a linear kernel because we have a 0.1 less precison but we less
# support vectors. This is important in classification time

clf = svm.SVC(kernel='linear', probability=False)
clf.fit(X_train, y_train)  ## classifier generated

# Save model to disk and also a vectorizer index
joblib.dump(clf, 'models/svm_model_tagger.pkl')
with open('models/vectorizer_tagger.pkl', 'wb') as o_file:
    pickle.dump(vectorizer, o_file)

# Dump info about the model
print("Supported vectors length: %s" % str(clf.support_vectors_.shape))
print("Dual coef. length: %s" % str(clf.dual_coef_.shape))

score = clf.score(X_test, y_test)
print("Score k-fold validation: %.4f%%" % round(score, 4))

# I love the equalizer in my new scotch walkman
# The cell batery works bad
# The hotel is in a great location close to all that downtown Portsmouth has to offer
# We had a shot of scotch whiskey at the hotel bar
sample = vectorizer.transform(['We had a shot of scotch whiskey at the hotel bar']).toarray()
print(sample.shape)
print(clf.predict(sample))

