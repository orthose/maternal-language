from utils import load_data
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
import estimators as est


# Entraînement sur train.txt et prédiction sur data.txt
# Les prédictions sont enregistrées dans output.txt


####################
### Entraînement ###
####################

df = load_data("../data/train.txt")
X, y = df["text"], df["language"]

pipe = make_pipeline(
    est.SpecialCharsCorrector(),
    est.NumbersTokenizer(),
    TfidfVectorizer(lowercase=False, ngram_range=(1, 2), 
                    tokenizer=est.nltk_tokenize, token_pattern=None),
    LinearSVC(random_state=42, max_iter=3000, C=2.)
)

pipe.fit(X, y)

##################
### Prédiction ###
##################

with open("../data/data.txt", 'r') as f:
    data = f.read()[:-1]
    
data = data.split('\n')
df = pd.DataFrame(data, columns=["text"])
X = df["text"]

y_pred = pipe.predict(X)

######################
### Enregistrement ###
######################

for i in range(len(data)):
    data[i] = f'({y_pred[i]}) '+data[i]
    
data = '\n'.join(data)

with open("../data/output.txt", 'w') as f:
    f.write(data)
