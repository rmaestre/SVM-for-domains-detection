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
import time

# <codecell>

# Load previus calculated SVM model
t_start = time.time()
clf = joblib.load('models/svm_model.pkl')
print("SVM model elaped time %.4f " % (round(time.time()-t_start,3)))
# Load vectorized index
t_vectorized_start = time.time()
with open('models/vectorizer.pkl', 'rb') as o_file:
    vectorizer = pickle.load(o_file)
print("Vectorized elaped time %.4f " % (round(time.time()-t_vectorized_start,3)))
print("Whole model elaped time %.4f " % (round(time.time()-t_start,3)))

# <codecell>

# I love the equalizer in my new scotch walkman
# The cell batery works bad
# The hotel is in a great location close to all that downtown Portsmouth has to offer
# We had a shot of scotch whiskey at the hotel bar
t_start = time.time()
sample = vectorizer.transform(['We had a shot of scotch whiskey at the hotel bar']).toarray()
label = clf.predict(sample)
print("Classification elaped time %.4f " % (round(time.time()-t_start,3)))
print("Class prediction: %s" % label)

# <codecell>


