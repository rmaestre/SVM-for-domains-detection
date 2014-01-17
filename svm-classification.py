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

# Lang options
langs = {0: "sp", 1: "en"}
lang = langs[1]

# Load previus calculated SVM model
t_start = time.time()
clf = joblib.load('models/%s/svm_model.pkl'%lang)
print("SVM model elaped time %.4f " % (round(time.time()-t_start,3)))

# Load vectorized index
t_vectorized_start = time.time()
with open('models/%s/vectorizer.pkl'%lang, 'rb') as i_file:
    vectorizer = pickle.load(i_file)

# Debug info
print("Vectorized elaped time %.4f " % (round(time.time()-t_vectorized_start,3)))
print("Whole model elaped time %.4f " % (round(time.time()-t_start,3)))

# <codecell>

# Set some samples
if lang == "en":
    samples = [("i love the equalizer in my new scotch walkman", 0), 
               ("the cell batery works bad", 0),
               ("we had a shot of scotch whiskey at the hotel bar",1), 
               ("the hotel is in a great location close to all that downtown portsmouth has to offer",1)]
elif lang == "sp":
    samples = [("El de 99 euros tiene 16Gb de memoria, el de 139 sólo 8Gb", 0), 
               ("Los amigos pueden compartir las pantallas así como explorar la música, archivos y juegos favoritos del otro", 0),
               ("El The Palm at Playa dispone de una terraza en la azotea con piscina, bar, spa y gimnasio",1), 
               ("ofrece 2 piscinas al aire libre, bañera de hidromasaje y habitaciones cómodas con balcón",1)]
    
# Classify each sample
for sample in samples:
    t_start = time.time()
    vector = vectorizer.transform([sample[0]]).toarray()
    label = clf.predict(vector)
    print("Labeled: %s Prediction: %s" % (sample[1], label))
    print("Elaped time %.6f\n" % (round(time.time()-t_start,5)))

# <codecell>


# <codecell>


