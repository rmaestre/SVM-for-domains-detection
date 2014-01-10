SVM model for domains detection
=========================

By means of a SVM model (linear kernel) we classify reviews between two specific domains: hotels and electronics.
The main details of the model in the follow stages are:

* **Trainning**:
  * Corpus with 10000 vectors, 5000 per each domain (10000x9024)
  * SVM with 1496 Supported vectors (1496x9024)
  * Label 0 for hotels and label 1 for electronics
  * K-fold validation (30%): 0.9443%
* **Classification**: In this stage, we load the previous saved models, therefore we have the next elapsed times to: 
  * Load SVM model: 0.0100 s.
  * Load Vectorized: 0.2520 s.
  * Load Total time: 0.2630  s.
  * Text classification: 0.0030 s.
  
Classification
--------------------
Firstly, we should load the Svm model
```python
# Load previus calculated SVM model
clf = joblib.load('models/svm_model.pkl')

# Load vectorized index
with open('models/vectorizer.pkl', 'rb') as i_file:
    vectorizer = pickle.load(i_file)
```
With the SVM model and the vectorized DS, we should convert the sample texts into a vector through vectorized DS
```
vector = vectorizer.transform("some text to classify").toarray()
```
later, we can predict the label by means of the calculated SVM model

```
label = clf.predict(vector)
```
