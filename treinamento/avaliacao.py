"""Metricas, selecao de limiar e intervalos de confianca por bootstrap."""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, average_precision_score, brier_score_loss, confusion_matrix,
    f1_score, precision_recall_curve, precision_score, recall_score, roc_auc_score,
)


METRICAS = ["Acuracia", "Precisao", "Recall", "F1", "AUC_ROC", "AP", "Brier"]


def selecionar_limiar(y_validacao, probabilidades):
    """Limiar que maximiza o F1 na validacao; em empate, usa o maior limiar."""
    precisao, recall, limiares = precision_recall_curve(y_validacao, probabilidades)
    denominador = precisao[:-1] + recall[:-1]
    f1 = np.divide(2 * precisao[:-1] * recall[:-1], denominador,
                   out=np.zeros_like(denominador), where=denominador > 0)
    indice = np.flatnonzero(f1 == f1.max())[-1]
    return float(limiares[indice]), float(f1[indice])


def calcular_metricas(y, probabilidades, limiar):
    previsao = np.asarray(probabilidades) >= limiar
    tn, fp, fn, tp = confusion_matrix(y, previsao, labels=[0, 1]).ravel()
    return {
        "Acuracia": accuracy_score(y, previsao),
        "Precisao": precision_score(y, previsao, zero_division=0),
        "Recall": recall_score(y, previsao, zero_division=0),
        "F1": f1_score(y, previsao, zero_division=0),
        "AUC_ROC": roc_auc_score(y, probabilidades),
        "AP": average_precision_score(y, probabilidades),
        "Brier": brier_score_loss(y, probabilidades),
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
    }


def selecionar_limiar_custo(y, probabilidades, custo_fn, custo_fp=1.0):
    """Limiar de menor custo total, dado o custo de um falso negativo."""
    y, p = np.asarray(y), np.asarray(probabilidades)
    ordem = np.argsort(-p, kind="stable")
    ys, ps = y[ordem], p[ordem]
    finais = np.r_[np.flatnonzero(np.diff(ps)), len(ps) - 1]
    tp = np.r_[0, np.cumsum(ys)[finais]]
    fp = np.r_[0, (finais + 1) - np.cumsum(ys)[finais]]
    limiares = np.r_[np.nextafter(ps[0], np.inf), ps[finais]]
    custos = custo_fn * (y.sum() - tp) + custo_fp * fp
    return float(limiares[np.argmin(custos)])


def bootstrap_pareado(previsoes, repeticoes=2000, seed=42):
    """Intervalos de 95% por bootstrap estratificado e pareado por registro.

    Modelos e limiares ficam fixos; a variacao de novos treinamentos nao e medida.
    """
    linhas = []
    pares = [("MLP_sem", "RL_sem"), ("MLP_smote_1", "MLP_sem"),
             ("RL_smote_1", "RL_sem"), ("MLP_smote_1", "RL_smote_1")]
    for cenario, dados in previsoes.groupby("Cenario", sort=False):
        modelos = {nome: g.sort_values("Registro").set_index("Registro")
                   for nome, g in dados.groupby("Modelo", sort=False)}
        nomes = list(modelos)
        referencia = modelos[nomes[0]]
        for tabela in modelos.values():
            if not tabela.index.equals(referencia.index) or not tabela.y.equals(referencia.y):
                raise ValueError("Bootstrap pareado exige os mesmos registros e alvos")
        y = referencia.y.to_numpy()
        positivos, negativos = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
        if not len(positivos) or not len(negativos):
            raise ValueError("Bootstrap estratificado exige ambas as classes")
        rng = np.random.default_rng(seed)
        amostras = {nome: np.empty((repeticoes, 3)) for nome in nomes}
        pontos = {}
        for nome, tabela in modelos.items():
            m = calcular_metricas(y, tabela.Probabilidade, tabela.Limiar.iloc[0])
            pontos[nome] = np.array([m[k] for k in ("F1", "AUC_ROC", "AP")])
        for i in range(repeticoes):
            indices = np.r_[rng.choice(negativos, len(negativos)), rng.choice(positivos, len(positivos))]
            yi = y[indices]
            for nome, tabela in modelos.items():
                p = tabela.Probabilidade.to_numpy()[indices]
                pred = p >= tabela.Limiar.iloc[0]
                tp = np.sum(pred & (yi == 1))
                f1 = 2 * tp / (pred.sum() + yi.sum())
                amostras[nome][i] = f1, roc_auc_score(yi, p), average_precision_score(yi, p)
        comparacoes = [(nome, None) for nome in nomes] + [
            (a, b) for a, b in pares if a in modelos and b in modelos]
        for a, b in comparacoes:
            valores = amostras[a] if b is None else amostras[a] - amostras[b]
            ponto = pontos[a] if b is None else pontos[a] - pontos[b]
            for j, metrica in enumerate(("F1", "AUC_ROC", "AP")):
                inferior, superior = np.quantile(valores[:, j], [0.025, 0.975])
                linhas.append({"Cenario": cenario, "Modelo_A": a, "Modelo_B": b or "",
                               "Metrica": metrica, "Estimativa": ponto[j],
                               "IC95_inferior": inferior, "IC95_superior": superior,
                               "Repeticoes": repeticoes,
                               "Metodo": "bootstrap_pareado_estratificado_condicional"})
    return pd.DataFrame(linhas)
