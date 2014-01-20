# -*- coding: utf-8 -*-
import pickle
import re

from sklearn import svm
from sklearn import cross_validation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.externals import joblib
from numpy import array

def insert_processed_review(corpus_result, labels_result, review, label, vectors_size, corpus_number_perdomain):
    """
    Review cleaning (punctuation).
    Being a review: "word1 word2 word3 word4 word5 word6" and vectors_size = 3
    2 vectors are extracted: ["word1 word2 word3","word4 word5 word6"]
    which are inserted in 'corpus_result' (and their corresponding label (the review one) in 'labels_result')
    """
#    exclude = set(string.punctuation).union(['¡', '¿', u'£', '€', '$'])  # Spanish
#    cleaned_review = ''.join(ch if ch not in exclude else ' ' for ch in review)  # list comprehension
#    words = cleaned_review.split()
    words = re.findall(r'\b[a-zA-ZáéíóúüÁÉÍÓÚÜ]+\b', line)
    if len(words) > 0:
        # Create vectors of length = vectors_size
        for i in range(0, len(words), vectors_size):
            if len(corpus_result) < corpus_number_perdomain:
                corpus_result.append(' '.join(words[i:i+vectors_size]))
                labels_result.append(label)

# Parameters to control the length of the vectors and the Matrix rows
corpus_number_perdomain = 40000
vectors_size = 20

langs = {0: "sp", 1: "en"}
lang = langs[0]

# DS to save corpus and labels
corpus = []
labels = []

domains = {"data/%s/electronics/electronics.txt"%lang: 0, "data/%s/hotels/hotels.txt"%lang: 1}
for file_name in domains:
    # Loading i-domain review-corpus
    with open(file_name, "r") as file_in:
        for line in file_in.readlines():  # 1 review per line!
            # Update corpus with new vectors
            insert_processed_review(corpus, labels, line, domains[file_name], vectors_size, corpus_number_perdomain)
            if len(corpus) == corpus_number_perdomain:
                break
    corpus_number_perdomain *= 2  # corpus/labels shared for all domains
            
# We need the same length for both DS
assert(len(corpus) == len(labels))
print("Corpus size: %d " % len(corpus))

# Vectorization: transforming text corpora to TF matrices
# http://scikit-learn.org/stable/modules/feature_extraction.html#common-vectorizer-usage
vectorizer = CountVectorizer(min_df=1)
X = vectorizer.fit_transform(corpus)  # vocabulary generated for this input corpus
#X.toarray()

# Transform list of labels to an array
y = array(labels)

print("Corpus matrix shape: %s " % str(X.shape))
print("Labels vector shape: %s " % str(y.shape))

# Training and validation data (k-fold = 30%)
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.3, random_state=1)


# With a RBF kernel, gamma=0.01, corpus_number_perdomain = 5000 and
# vectors_size = 20 to reach the maximum score, however we
# use a linear kernel because we have a 0.1 less precison but we less
# support vectors. This is important in classification time

clf = svm.SVC(kernel='linear', probability=False)
clf.fit(X_train, y_train)  ## classifier generated

# Save model to disk and also a vectorizer index
joblib.dump(clf, 'models/%s/svm_model.pkl'%lang)
with open('models/%s/vectorizer.pkl'%lang, 'wb') as o_file:
    pickle.dump(vectorizer, o_file)

# Dump info about the model
print("\nSupported vectors length: %s" % str(clf.support_vectors_.shape))
print("Dual coef. length: %s" % str(clf.dual_coef_.shape))

score = clf.score(X_test, y_test)
print("\nScore k-fold validation: %.4f%%" % round(score, 4))

# I love the equalizer in my new scotch walkman
# The cell batery works bad
# The hotel is in a great location close to all that downtown Portsmouth has to offer
# We had a shot of scotch whiskey at the hotel bar
sample = vectorizer.transform(['We had a shot of scotch whiskey at the hotel bar']).toarray()
print(sample.shape)
print(clf.predict(sample))

