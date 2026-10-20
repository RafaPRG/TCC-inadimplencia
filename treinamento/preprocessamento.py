import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder


COLUNAS_CATEGORICAS = ["gender", "owns_car", "owns_house", "occupation_type", "migrant_worker"]


def criar_preprocessador(X_treino):
    """Imputacao, escala MinMax e one-hot; deve ser ajustado so nos dados de ajuste."""
    categoricas = [c for c in COLUNAS_CATEGORICAS if c in X_treino.columns]
    numericas = [c for c in X_treino.columns if c not in categoricas]
    numerico = Pipeline([
        ("imputacao", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("escala", MinMaxScaler()),
    ])
    categorico = Pipeline([
        ("imputacao", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
        ("codificacao", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("numericas", numerico, numericas),
        ("categoricas", categorico, categoricas),
    ], sparse_threshold=0)
def aplicar_smote_fold(X_train, y_train, random_state=42, sampling_strategy=1.0):
    """Aplica SMOTE apenas aos dados de ajuste de uma divisao."""
    classes, contagens = np.unique(y_train, return_counts=True)
    if len(classes) != 2 or contagens.min() < 2:
        raise ValueError("SMOTE exige duas classes e ao menos dois exemplos minoritarios")
    if not 0 < sampling_strategy <= 1:
        raise ValueError("Proporcao SMOTE deve estar em (0, 1]")
    if int(contagens.max() * sampling_strategy) <= contagens.min():
        return np.array(X_train, copy=True), np.array(y_train, copy=True)
    smote = SMOTE(random_state=random_state, sampling_strategy=sampling_strategy,
                  k_neighbors=min(5, int(contagens.min()) - 1))
    X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
    return X_train_bal, y_train_bal
