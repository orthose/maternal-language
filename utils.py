import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
from seaborn import heatmap
from typing import List, Tuple


def load_data(file: str) -> pd.DataFrame:
    """
    Chargement du dataset
    
    :param file: chemin du fichier
    :return: dataframe pandas
    """
    # Chargement du dataset
    with open(file, 'r') as f:
        data = f.read()[:-1] # On supprime la dernière ligne

    # Découpage en liste de textes
    data = data.split('\n')
    # Récupération de l'étiquette
    data = [(t[0:5][1:-1], t[5:]) for t in data]

    # Conversion vers pandas dataframe
    df = pd.DataFrame(data, columns=["language", "text"])
    
    return df

def train_valid_test_split(df: pd.DataFrame, size: Tuple[int] = (600, 200, 100)
                          ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Découpe le dataset en trois morceaux en précisant le nombre d'observations par langue
    
    :param df: dataframe à découper
    :param size: tuple de taille 3 dont la somme fait 900
    :return: tuple de taille 3 avec les dataframes train, valid et test
    """
    assert sum(size) == 900
    df["rank"] = df.sample(frac=1, random_state=42).groupby("language")["language"].rank(method="first", ascending=True)
    df_train = df[df["rank"] <= size[0]].drop("rank", axis=1)
    df_valid = df[(size[0] < df["rank"]) & (df["rank"] <= size[0] + size[1])].drop("rank", axis=1)
    df_test = df[(size[0] + size[1] < df["rank"]) & (df["rank"] <= size[0] + size[1] + size[2])].drop("rank", axis=1)
    
    return df_train, df_valid, df_test

def evaluate(y_true: np.array, y_pred: np.array, labels: List[str] = None):
    """
    Évalue les prédictions du modèle en affichant:
    accuracy, f1-score, matrice de confusion
    
    :param y_true: étiquettes réelles 
    :param y_pred: étiquettes prédites
    :param labels: Étiquettes à afficher dans la matrice de confusion
    """
    print(f"accuracy: {accuracy_score(y_true, y_pred)}")
    print(f"f1-score: {f1_score(y_true, y_pred, average='macro')}")
    if labels is None:
        labels = ['TEL', 'HIN', 'CHI', 'KOR', 'JPN', 'FRE', 'SPA', 'ITA', 'TUR', 'ARA', 'GER']
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    heatmap(matrix, annot=True, fmt='.3g', xticklabels=labels, yticklabels=labels)
    plt.yticks(rotation=0)
    plt.show()
    
class PipelineEvaluator:
    """
    Classe permettant d'évaluer des pipelines pour trouver la meilleure
    """
    def __init__(self, X_train, y_train, X_valid, y_valid):
        """
        :param X_train: ensemble d'entraînement
        :param y_train: étiquettes associées
        """
        self.X_train = X_train
        self.y_train = y_train
        self.X_valid = X_valid
        self.y_valid = y_valid
        self.pipelines = [(Pipeline(steps=[]), 0.)]
        
    def evaluate(self, pipe: Pipeline) -> float:
        """
        Évalue une pipeline sur l'ensemble d'entraînement 
        et renvoie le f1-score associé sur l'ensemble de validation
        
        :param pipe: pipeline à évaluer
        :return: f1-score sur l'ensemble de validation
        """
        pipe.fit(self.X_train, self.y_train)
        y_pred = pipe.predict(self.X_valid)
        score = f1_score(self.y_valid, y_pred, average="macro")
        best = max(self.pipelines, key=lambda x: x[1])[1]
        self.pipelines.append((pipe, score))
        
        # Est-ce la meilleure pipeline ?
        if score >= best:
            print("!!! BEST !!!")
        
        return score
    
    def best(self) -> Tuple[Pipeline, float]:
        """
        Trouve la meilleure pipeline avec le score le plus élevé
        
        :return: pipeline, score
        """
        return max(self.pipelines, key=lambda x: x[1])
