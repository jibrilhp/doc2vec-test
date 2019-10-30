from flask import request, url_for,jsonify
from flask_api import FlaskAPI, status, exceptions
import numpy as np
import math
import pandas as pd
import re
import os
import json
import sys
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



app = FlaskAPI(__name__)

@app.route('/example/')
def example():
    return {'hello': 'world'}

@app.route('/spmi/api/<perintah>',methods=['POST','OPTIONS'])
def api(perintah):
    #0 lihat antrean ada atau nggak
    if (perintah == '0'):
        if (os.path.isfile('wait.txt')):
            f_wait = open("wait.txt","r")
            if (f_wait.read() != 'hmm'):
                data = {'status':'ok','message':'STATUS_SIAP_BERJALAN'}
                return jsonify(data)
            else:
                data = {'status':'ok','message':'STATUS_SIBUK'}
                return jsonify(data)
        else:
            data = {'status':'ok','message':'STATUS_TIDAK_TERSEDIA'}
            return jsonify(data)

    #1 menganalisis data
    if (perintah == '1'):
        f_wait = open("wait.txt","w")
        f_wait.write('hmm')
        f_wait.close()

        data = request.get_json(force=True)
        
        if (data != ''):
            hasil =  json.dumps(data)
            parsed = json.loads(str(hasil))
            teks = parsed['teks_analysis']

            if os.path.isfile("hasil.txt"):
                f_wait = open("hasil.txt","r")
                hasil =  f_wait.read()
                f_wait.close()

                os.remove("hasil.txt")

                return hasil
            else:            
                hasil_periksa = periksa_teks(text=teks)
                f_wait = open("hasil.txt","w")
                f_wait.write(hasil_periksa)
                f_wait.close()

                f_wait2 = open("wait.txt","w")
                f_wait2.write('done')
                f_wait2.close()
           


            
            return hasil_periksa
        else:
            return 'data tidak ada..'

        
    #3 Reset antrean (entah masih ada proses atau nggak)
    if (perintah == '3'):
        f_wait = open("wait.txt","w")
        f_wait.write('done')
        f_wait.close()

        
def vector_for_learning(model, input_docs):
    sents = input_docs
    targets, feature_vectors = zip(*[(doc.tags[0], model.infer_vector(doc.words, steps=20)) for doc in sents])
    return targets, feature_vectors 

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

#path_dm = os.path.abspath('model_dm0/')



def vector_for_learning(model, input_docs):
    sents = input_docs
    targets, feature_vectors = zip(*[(doc.tags[0], model.infer_vector(doc.words, steps=20)) for doc in sents])
    return targets, feature_vectors 

# fname = get_tmpfile("./cobaa.d2v")
def periksa_teks (text):
    
    model_dbow = Doc2Vec.load("cobaa2.d2v")

    train_documents = []
    test_documents = []
    i = 0

    tags_index = {'Nilai ke-1': 1 , 'Nilai ke-2': 2, 'Nilai ke-3': 3, 'Nilai ke-4': 4, 'Nilai ke-5': 5, 'Nilai ke-6': 6, 'Nilai ke-7': 7}
    isian_array = {}


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

    for xx in range(1,8):
        print('loading >> ' + str(xx), file=sys.stderr)
        sys.stdout.flush()
        train_documents  = utils.shuffle(train_documents)
        test_documents.append( TaggedDocument(words=word_tokenizer(pre_processing(text)), tags=[tags_index.get("Nilai ke-" + str(xx), 8)] ))

        y_train, X_train = vector_for_learning(model_dbow, train_documents)
        y_test, X_test = vector_for_learning(model_dbow, test_documents)



        logreg = LogisticRegression(n_jobs=1, C=1e5)
        logreg.fit(X_train, y_train)
        y_pred = logreg.predict(X_test)

        hitung = f1_score(y_test, y_pred, average='weighted', labels=np.unique(y_pred))
    
        if (hitung > 0):
            isian_array = {"berhitung":"1","nilai":xx}
            return (json.dumps(isian_array))
            break

    print("sudah dibaca")
    
    
    
    if (hitung > 0):
        return 0
    else:
        test_data = word_tokenizer(pre_processing(text).lower())
        
        infer_vector = model_dbow.infer_vector(test_data)
        similar_documents = model_dbow.docvecs.most_similar([infer_vector])
        isian_array = {"berhitung":"2","nilai": pd.Series(similar_documents).to_json(orient='values')}
        return json.dumps(isian_array)


if __name__ == "__main__":
    website_url = 'localhost:1311'
    app.config['SERVER_NAME'] = website_url 
    app.run(debug=False)