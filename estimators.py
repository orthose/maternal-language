import pandas as pd
import numpy as np
from statistics import mean
from sklearn.base import BaseEstimator
from gensim.models import Word2Vec
from gensim.utils import tokenize
from nltk.tokenize import word_tokenize
from sklearn.svm import LinearSVC
from typing import List
import re


#################
### Nettoyage ###
#################

def correct_special_chars(doc: str) -> str:
    # Harmonisation de la ponctuation
    doc = re.sub(r"[\[{]", "(", doc)
    doc = re.sub(r"[\]}]", ")", doc)
    
    # Suppression des caractères spéciaux
    doc = re.sub(r"[+*=#~^|_\\'`´]", "", doc)
    
    return doc

class SpecialCharsCorrector(BaseEstimator):
    def fit(self, X, y=None): 
        return self
    
    def transform(self, X, y=None):
        return X.apply(correct_special_chars)

####################
### Tokenization ###
####################
    
def tokenize_numbers(doc: str) -> str:
    return re.sub("\d+", "#NUMBER", doc)

class NumbersTokenizer(BaseEstimator):
    def fit(self, X, y=None): 
        return self
    
    def transform(self, X, y=None):
        return X.apply(tokenize_numbers)
    
def gensim_tokenize(doc: str) -> List[str]:
    return list(tokenize(doc))

def nltk_tokenize(doc: str) -> List[str]:
    return word_tokenize(doc)
        
class SimpleTokenizer(BaseEstimator):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        return X.apply(lambda x: x.split())
    
#####################
### Vectorisation ###
#####################
    
class Word2VecEstimator(BaseEstimator):
    def __init__(self, **kwargs):
        self.w2v = Word2Vec(**kwargs)
        
    def fit(self, X, y=None):
        self.w2v.build_vocab(X)
        return self
        
    def transform(self, X, y=None):
        X = [np.array([self.w2v.wv[w] for w in doc if w in self.w2v.wv]) for doc in X]
        X = np.array([doc.mean(axis=0) for doc in X])
        return X
    
################
### Features ###
################

def count_words(doc: str) -> int:
    return len([w for w in re.split(r"[ .,:!?']", doc) if w != ''])    

class WordsCounter(BaseEstimator):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        return X.apply(count_words).to_numpy().reshape(-1, 1)

def mean_chars_sentence(doc: str) -> float:
    sentences = re.split(r'[.!?]+', doc)
    if sentences[-1] == '':
        sentences.pop(-1)
    return mean([len(s) for s in sentences])
    
class MeanCharsSentence(BaseEstimator):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        return X.apply(mean_chars_sentence).to_numpy().reshape(-1, 1)
    
def mean_words_sentence(doc: str) -> float:
    count_words = []
    sentences = re.split(r'[.!?]+', doc)
    if sentences[-1] == '':
            sentences.pop(-1)
    for snt in sentences:
        words = re.split(r"[ ,':]", snt)
        count_words.append(len([w for w in words if w != '']))
    return mean(count_words)

class MeanWordsSentence(BaseEstimator):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        return X.apply(mean_words_sentence).to_numpy().reshape(-1, 1)

def lexical_richness(doc: str) -> int:
    tokens = [w for w in re.split(r"[ .,:!?']", doc) if w != '']
    n_types = len(set(tokens))
    return n_types

class LexicalRichness(BaseEstimator):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        return X.apply(lexical_richness).to_numpy().reshape(-1, 1)
    
def count_double_consonant(doc: str) -> int:
    consonant = list('abcdfghjklmnpqrstvwxyz')
    regex = '|'.join([c+c for c in consonant])
    return len(re.findall(regex, doc))

class DoubleConsonantCounter(BaseEstimator):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        return X.apply(count_double_consonant).to_numpy().reshape(-1, 1)
    
def count_contractions(doc: str) -> int:
    regex = r"['’][ ]*(m|re|s|ve|ll|d|t)"
    return len(re.findall(regex, doc))

class ContractionsCounter(BaseEstimator):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        return X.apply(count_contractions).to_numpy().reshape(-1, 1)
    
##########################
### Cascade de Modèles ###
##########################

def build_metalabel(y: pd.Series, labels: List[str]) -> pd.Series:
    metalabel = '-'.join(labels)
    return y.apply(lambda x: metalabel if x in labels else x)

class CascadeModel(BaseEstimator):
    def __init__(self, **kwargs):
        """
        Les dictionnaire des arguments est utilisé 
        pour les hyperparamètres du modèle principal.
        """
        # Modèle principal
        self.master = LinearSVC(**kwargs)
        
        # Modèle (TEL,HIN)
        self.telhin = LinearSVC(max_iter=500)
        
        # Modèle (KOR,JPN)
        self.korjpn = LinearSVC(max_iter=500)
        
        # Modèle (FRE,SPA)
        self.frespa = LinearSVC(max_iter=500)
       
    @staticmethod
    def build_labels(y: pd.Series) -> pd.Series:
        y = build_metalabel(y, ['TEL', 'HIN'])
        y = build_metalabel(y, ['KOR', 'JPN'])
        y = build_metalabel(y, ['FRE', 'SPA'])
        return y
        
    def fit(self, X, y) -> 'CascadeModel':
        # Regroupement des paires d'étiquettes
        y_meta = CascadeModel.build_labels(y)
        
        # Entraînement du modèle principal
        self.master.fit(X, y_meta)
        
        # Entraînement du modèle (TEL,HIN)
        mask = (y_meta == 'TEL-HIN')
        X_t, y_t = X[mask], y[mask]
        self.telhin.fit(X_t, y_t)
        
        # Entraînement du modèle (KOR,JPN)
        mask = (y_meta == 'KOR-JPN')
        X_t, y_t = X[mask], y[mask]
        self.korjpn.fit(X_t, y_t)
        
        # Entraînement du modèle (FRE,SPA)
        mask = (y_meta == 'FRE-SPA')
        X_t, y_t = X[mask], y[mask]
        self.frespa.fit(X_t, y_t)
        
    def predict(self, X) -> np.array:
        y_pred = self.master.predict(X)
        for i in range(X.shape[0]):
            if 'TEL-HIN' == y_pred[i]:
                y_pred[i] = self.telhin.predict(X[i])[0]
                
            elif 'KOR-JPN' == y_pred[i]:
                y_pred[i] = self.korjpn.predict(X[i])[0]
                
            elif 'FRE-SPA' == y_pred[i]:
                y_pred[i] = self.frespa.predict(X[i])[0]
                
        return y_pred
