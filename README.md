SVM model for domains detection
=========================

By means of a SVM model (linear kernel) we classify reviews between two specific domains: hotels and electronics.
The main details of the model in the follow stages are:

* **Trainning**:
  * Corpus with 80000 vectors, 40000 per each domain (80000, 23789)
  * SVM with 6833 Supported vectors (6833, 23789)
  * Label 0 for hotels and label 1 for electronics
  * K-fold validation (30%): 0.9573%
* **Classification**: In this stage, we load the previous saved models, therefore we have the next elapsed times to: 
  * SVM model elapsed time 0.0060 s.
  * Vectorized elapsed time 0.1760 s. 
  * Whole model elapsed time 0.1830 s. 
  * Text classification: 0.002480 s.
  
Classification
--------------------
Firstly, we should load the Svm model
```python
# Load previus calculated SVM model with a specific language
clf = joblib.load('models/en/svm_model.pkl')

# Load vectorized index with a specific language
with open('models/en/vectorizer.pkl', 'rb') as i_file:
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
