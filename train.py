import numpy as np
import math
import pandas as pd
import re
import os
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.test.utils import common_texts, get_tmpfile
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import utils
import csv
#from tqdm import tqdm
#import multiprocessing
#from nltk.tokenize import word_tokenize
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory, StopWordRemover, ArrayDictionary
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

#path_dm = os.path.abspath('model_dm0/')

def pre_processing(text):
    stopwords = pd.read_csv('stopwordbahasa.csv', names=['stopword'])['stopword'].tolist()

    stem = StemmerFactory() 
    stemmer = stem.create_stemmer()
    factory = StopWordRemoverFactory()
    stopword = StopWordRemover(ArrayDictionary(factory.get_stop_words() + stopwords))

    clean_str = text.lower() # lowercase
    clean_str = re.sub(r"(?:\@|#|https?\://)\S+", " ", clean_str) # eliminate username, url, hashtags
    clean_str = re.sub(r'&amp;', '', clean_str) # remove &amp; as it equals &
    clean_str = re.sub(r'[^\w\s]',' ', clean_str) # remove punctuation
    clean_str = re.sub('[\s\n\t\r]+', ' ', clean_str) # remove extra space
    clean_str = clean_str.strip() # trim
    clean_str = " ".join([stemmer.stem(word) for word in clean_str.split()]) # stem
    clean_str = stopword.remove(clean_str) # remove stopwords
    return clean_str

def word_tokenizer(text):
    #tokenizes and stems the text
    tokens = word_tokenize(text)
    fac2 = StemmerFactory()
    stemmer = fac2.create_stemmer()
    factory = StopWordRemoverFactory()
    tokens = [stemmer.stem(t) for t in tokens if t not in factory.get_stop_words()]
    
    return tokens

def vector_for_learning(model, input_docs):
    sents = input_docs
    targets, feature_vectors = zip(*[(doc.tags[0], model.infer_vector(doc.words, steps=20)) for doc in sents])
    return targets, feature_vectors 

# fname = get_tmpfile("./cobaa.d2v")


model_dbow = Doc2Vec.load("cobaa2.d2v")

train_documents = []
test_documents = []
i = 0

tags_index = {'Nilai ke-1': 1 , 'Nilai ke-2': 2, 'Nilai ke-3': 3, 'Nilai ke-4': 4, 'Nilai ke-5': 5, 'Nilai ke-6': 6, 'Nilai ke-7': 7}


FILEPATH = 'gabungan.csv'
with open(FILEPATH, 'r') as csvfile:
    with open('gabungan.csv', 'r',encoding='ISO-8859-1') as csvfile:
        spmi = csv.reader(csvfile, delimiter=';', quotechar='"')
        for row in spmi:
            if i == 0:
                i += 1
                continue
            i += 1
            if i <= 1028:            
                train_documents.append(TaggedDocument(words=word_tokenizer(pre_processing(str(row[1]))), tags=[tags_index.get(row[2], 8)] ))

        
train_documents  = utils.shuffle(train_documents)
test_documents.append( TaggedDocument(words=word_tokenizer(pre_processing(" 0%-39%   dilengkapi dengan deskripsi matakuliah, RPKPS/Sillabus, Modul dan SAP.")), tags=[tags_index.get("Nilai ke-1", 8)] ))

y_train, X_train = vector_for_learning(model_dbow, train_documents)
y_test, X_test = vector_for_learning(model_dbow, test_documents)



logreg = LogisticRegression(n_jobs=1, C=1e5)
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
print('Akurasi untuk uji similaritas SPMI %s' % accuracy_score(y_test, y_pred))
print('F1 score uji similaritas SPMI: {}'.format(f1_score(y_test, y_pred, average='weighted')))


