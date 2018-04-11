import gensim
import numpy as np
import re
import pickle
def read_word2vec(file_path):
    model =gensim.models.KeyedVectors.load_word2vec_format(file_path)
    return model
def text2img(text,word2vecmodel,size=40):
    img=[]
    words=text.split()
    for w in words:
        if(w in word2vecmodel.vocab):
            img.append(word2vecmodel[w])
            if(len(img)==size):
                break
    while(len(img)<size):
        img.append([0 for i in range(0,200)])
    return np.asarray(img).T.flatten()
def process_text(text):
    text=text.lower()
    text=re.sub("[^a-z]"," ",text)
    return text
def load_data(filename):
    with open(filename,'rb') as f:
        data=pickle.load(f)
    return data