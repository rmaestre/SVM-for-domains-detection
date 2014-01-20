# -*- coding: utf-8 -*-
import pickle
import time

from sklearn.externals import joblib

# Lang options
langs = {0: "sp", 1: "en"}
lang = langs[1]

# Load previous calculated SVM model
t_start = time.time()
clf = joblib.load('models/%s/svm_model.pkl'%lang)
print("SVM model elapsed time %.4f " % (round(time.time()-t_start,3)))

# Load vectorized index
t_vectorized_start = time.time()
with open('models/%s/vectorizer.pkl'%lang, 'rb') as i_file:
    vectorizer = pickle.load(i_file)

# Debug info
print("Vectorized elaped time %.4f " % (round(time.time()-t_vectorized_start,3)))
print("Whole model elaped time %.4f " % (round(time.time()-t_start,3)))

# Set some samples
if lang == "en":
    samples = [("i love the equalizer in my new scotch walkman", 0), 
               ("the cell batery works bad", 0),
               ("The T bar connector is good quality", 0),
               ("The vendor was prompt and informative re despatch", 0),
               ("Good Phone if all you want to do is text + phone", 0),
               ("not like some phones where you need to charge every other day.", 0),
               ("the zoom stopped working as well", 0),
               ("What might have been a average stay was great because of her people skills", 1),
               ("Good to stay in town for few days. Parking inside the building", 1),
               ("Staff member Rebbeca was so helpfull, thoughtfull and pleasent", 1),
               ("so carrying my luggage down was not easy", 1),
               ("we had a shot of scotch whiskey at the hotel bar",1), 
               ("the hotel is in a great location close to all that downtown portsmouth has to offer",1)]
elif lang == "sp":
    samples = [("El de 99 euros tiene 16Gb de memoria, el de 139 sólo 8Gb", 0), 
               ("Los amigos pueden compartir las pantallas así como explorar la música, archivos y juegos favoritos del otro", 0),
               ("la nueva interfaz Metro de Windows 8 es perfecta para trabajar con varias aplicaciones a la vez",0),
               ("La unidad SSD 840 PRO de Samsung mejora la velocidad de respuesta de tu ordenador ",0),
               ("mi sorpresa fue que el PC funcionó perfectamente",0),
               ("Intel Core2 Quad CPU Q6600 @ 2.40GHz, con 3GB ",0),
               ("La comida es fabulosa y si no hay un restaurante abierto hay otro",1),
               ("Me lleve una buena impresion del lugar bonito limpio",1),
               ("todo muy limpio, la atención de todo el personal muy grata",1),
               ("Ah y no di ni una sola propina y me atendieron muy bien en el hotel",1),
               ("El The Palm at Playa dispone de una terraza en la azotea con piscina, bar, spa y gimnasio",1), 
               ("bañera de hidromasaje y habitaciones cómodas con balcón",1)]
    
# Classify each sample
for sample in samples:
    t_start = time.time()
    vector = vectorizer.transform([sample[0]]).toarray()
    label = clf.predict(vector)
    print("Labeled: %s Prediction: %s" % (sample[1], label))
    print("Elapsed time %.6f\n" % (round(time.time()-t_start,5)))
