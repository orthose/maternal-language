from sklearn.base import BaseEstimator
import re


def correct_special_chars(doc: str) -> str:
    # Harmonisation de la ponctuation
    doc = re.sub(r"[\[{]", "(", doc)
    doc = re.sub(r"[\]}]", ")", doc)
    
    # Suppression des caractères spéciaux
    doc = re.sub(r"[+*=#~^|_\\'`´]", "", doc)
    
    return doc

class CorrectSpecialChars(BaseEstimator):
    def fit(self, X, y=None): 
        return self
    
    def transform(self, X, y=None):
        return X.apply(correct_special_chars)
    
def tokenize_numbers(doc: str) -> str:
    return re.sub("\d+", "#NUMBER", doc)

class TokenizeNumbers(BaseEstimator):
    def fit(self, X, y=None): 
        return self
    
    def transform(self, X, y=None):
        return X.apply(tokenize_numbers)
