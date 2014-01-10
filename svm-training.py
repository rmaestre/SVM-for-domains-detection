# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

"""
Paradigma labs 2014
"""

# <codecell>

from sklearn import svm
from sklearn import cross_validation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.externals import joblib

import pickle
import re

# <codecell>

def insert_vectorized_line(corpus, labels, line, label, vectors_size, corpus_number_perdomain):
    """
    From a line: "word1 word2 word3 word4 word5 word6" and vectors_size = 3
    return     : ["word1 word2 word3","word4 word5 word6"]
    
    The return DS is the format to vectorizer (fit_transform) list of words into vectors
    """
    # Get list of words from the line ["w1", "w2" .... ]
    words =  re.findall(r'\b[a-z]+\b', line)
    if len(words) > 0:
        # Create vectors of length = vectors_size
        for i in range(0, len(words), vectors_size):
            if len(corpus) < corpus_number_perdomain:
                # Append line with vectors_size words and the label class
                corpus.append(' '.join(words[i:i+vectors_size]))
                labels.append(label)

# <codecell>

# Parameters to control the length of the vectors and the Matrix rows
corpus_number_perdomain = 5000
vectors_size = 20

# DS to save corpus and labels
corpus = []
labels = []

# Load corpus from domains review
domains = {"data/electronics/electronics.txt":0, "data/hotels/hotels.txt":1}
for file_name in domains:
    # Load corpus from hotels review
    with open(file_name, "r") as file_in:
        for line in file_in.readlines():
            # Update corpus with new vectors
            insert_vectorized_line(corpus, labels, line, domains[file_name], vectors_size, corpus_number_perdomain)
            if len(corpus) == corpus_number_perdomain:
                break
    corpus_number_perdomain *= 2
            
# We need the same length for Matrix and labels
assert(len(corpus) == len(labels))
print("Corpus lenght: %d " % len(corpus))

# Transform corpus of vectors of words into matrix
vectorizer = CountVectorizer(min_df=1)
X = vectorizer.fit_transform(corpus)
X.toarray()

# Transform list of labels into array
y = array(labels)

print("Corpus matrix shape: %s " % str(X.shape))
print("Labels vector shape: %s " % str(y.shape))

# <codecell>

# Training and validation data (k-fold)
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.3, random_state=1)


# With a RBF kernel, gamma=0.01, corpus_number_perdomain = 5000 and
# vectors_size = 20 to reach the maximun score, however we
# use a linear kernel because we have a 0.1 less preccison but we less 
# support vectors. This is important in classification time

clf = svm.SVC(kernel='linear', probability=False)

# Clasify
clf.fit(X_train, y_train) 

# Save model to disk and also a vectorizer index
joblib.dump(clf, 'models/svm_model.pkl')
with open('models/vectorizer.pkl', 'wb') as o_file:
    pickle.dump(vectorizer, o_file)

# Dump info about the model
print("\nSupported vectors length: %s" % str(clf.support_vectors_.shape))
print("Dual coef. length: %s" % str(clf.dual_coef_.shape))

# <codecell>

score = clf.score(X_test, y_test)
print("\nScore k-fold validation: %.4f%%" % round(score, 4))

# <codecell>

# I love the equalizer in my new scotch walkman
# The cell batery works bad
# The hotel is in a great location close to all that downtown Portsmouth has to offer
# We had a shot of scotch whiskey at the hotel bar
sample = vectorizer.transform(['We had a shot of scotch whiskey at the hotel bar']).toarray()
print(sample.shape)
print(clf.predict(sample))

# <codecell>


