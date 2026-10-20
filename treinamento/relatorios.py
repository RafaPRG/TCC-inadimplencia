"""Tabelas, figuras e relatorio de cada execucao."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import ConfusionMatrixDisplay, precision_recall_curve, roc_curve

from treinamento.avaliacao import METRICAS, bootstrap_pareado


PRINCIPAIS = ["MLP_sem", "RL_sem", "MLP_smote_1", "RL_smote_1"]


def tabela_markdown(df):
    linhas = ["| " + " | ".join(df.columns) + " |",
              "| " + " | ".join(["---"] * len(df.columns)) + " |"]
    for valores in df.itertuples(index=False, name=None):
        linhas.append("| " + " | ".join(str(v) for v in valores) + " |")
    return "\n".join(linhas)


def salvar_curvas(dados, pasta, cenario, smoke, extras):
    titulo = " [EXECUCAO RAPIDA]" if smoke else ""
    for nome, selecionados in [("principais", PRINCIPAIS), ("todos", dados.Modelo.unique())]:
        fig, eixos = plt.subplots(1, 2, figsize=(13, 6))
        for modelo in selecionados:
            g = dados[dados.Modelo == modelo]
            if g.empty:
                continue
            fpr, tpr, _ = roc_curve(g.y, g.Probabilidade)
            precision, recall, _ = precision_recall_curve(g.y, g.Probabilidade)
            eixos[0].plot(fpr, tpr, label=modelo)
            eixos[1].step(recall, precision, where="post", label=modelo)
        eixos[0].plot([0, 1], [0, 1], "--", color="gray")
        prevalencia = dados.drop_duplicates("Registro").y.mean()
        eixos[1].axhline(prevalencia, linestyle="--", color="gray", label=f"Prevalencia: {prevalencia:.3f}")
        eixos[0].set(xlabel="Taxa de falsos positivos", ylabel="Sensibilidade", title="ROC")
        eixos[1].set(xlabel="Recall", ylabel="Precisao", title="Precisao-recall")
        for eixo in eixos:
            eixo.set_xlim(-0.01, 1.01)
            eixo.set_ylim(-0.01, 1.01)
            eixo.grid(alpha=0.2)
        handles, labels = eixos[1].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=8)
        fig.suptitle(f"Teste final: {cenario}{titulo}")
        fig.tight_layout(rect=(0, 0.2 if nome == "todos" else 0.12, 1, 0.96))
        fig.savefig(pasta / f"roc_pr_{cenario}_{nome}.png", dpi=160)
        plt.close(fig)
    if extras:
        fig, eixo = plt.subplots(figsize=(7, 5))
        eixo.plot([0, 1], [0, 1], "--", color="gray")
        for modelo in PRINCIPAIS:
            g = dados[dados.Modelo == modelo]
            if not g.empty:
                positivos, probabilidades = calibration_curve(g.y, g.Probabilidade, n_bins=10, strategy="quantile")
                eixo.plot(probabilidades, positivos, marker="o", label=modelo)
        eixo.set(xlabel="Probabilidade prevista media", ylabel="Fracao de positivos",
                 title=f"Calibracao: {cenario}{titulo}")
        eixo.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(pasta / f"calibracao_{cenario}.png", dpi=160)
        plt.close(fig)


def gerar_relatorios(pasta, repeticoes, seed, smoke=False, extras=False):
    metricas = pd.read_csv(pasta / "metricas.csv")
    previsoes = pd.read_csv(pasta / "previsoes.csv")
    cv = metricas[metricas.Etapa == "cv"]
    teste = metricas[metricas.Etapa == "teste"]
    resumo = cv.groupby(["Cenario", "Modelo"])[METRICAS].agg(["mean", "std", "count"])
    resumo.columns = [f"{m}_{estatistica}" for m, estatistica in resumo.columns]
    resumo = resumo.reset_index()
    resumo.to_csv(pasta / "resumo_cv.csv", index=False)
    teste.to_csv(pasta / "teste_final.csv", index=False)
    metricas[["Cenario", "Etapa", "Fold", "Modelo", "TN", "FP", "FN", "TP"]].to_csv(
        pasta / "matrizes_confusao.csv", index=False)
    previsoes_teste = previsoes[previsoes.Etapa == "teste"]
    intervalos = bootstrap_pareado(previsoes_teste, repeticoes, seed)
    intervalos.to_csv(pasta / "intervalos_bootstrap.csv", index=False)
    figuras = pasta / "figuras"
    figuras.mkdir(exist_ok=True)
    texto = ["# Relatorio de execucao", "",
             "Execucao rapida: subamostra e treinamento reduzido; os numeros nao representam o experimento." if smoke
             else "Resultados do protocolo com selecao interna e teste reservado.", "",
             "Media +/- desvio padrao: folds externos do desenvolvimento. Cada fold seleciona "
             "configuracao/limiar na sua validacao interna. O desvio nao e erro padrao; os treinos se sobrepoem.", "",
             "AP e average precision, nao a integral trapezoidal da curva PR. "
             "Bootstrap: IC95% percentil, estratificado e pareado por registro do teste. "
             "Os ICs sao condicionais aos modelos treinados e nao incluem variabilidade de retreinamento; "
             "sao pontuais e nao ajustados para comparacoes multiplas.", "",
             "A base ja foi explorada anteriormente. Estes resultados nao constituem validacao externa inedita. "
             "Uma diferenca pequena de F1 nao demonstra superioridade por si so.", ""]
    for cenario, dados in previsoes_teste.groupby("Cenario", sort=False):
        salvar_curvas(dados, figuras, cenario, smoke, extras)
        rc = resumo[resumo.Cenario == cenario]
        tabela = pd.DataFrame({"Modelo": rc.Modelo})
        for metrica in ["F1", "AP", "AUC_ROC", "Precisao", "Recall"]:
            tabela[metrica] = [f"{m:.4f} +/- {s:.4f}" for m, s in
                               zip(rc[f"{metrica}_mean"], rc[f"{metrica}_std"])]
        texto += [f"## {cenario}: validacao cruzada", "", tabela_markdown(tabela), ""]
        tc = teste[teste.Cenario == cenario]
        tab_teste = tc[["Modelo", "Limiar", "F1", "AP", "AUC_ROC", "Precisao", "Recall"]].copy()
        for coluna in tab_teste.columns[1:]:
            tab_teste[coluna] = tab_teste[coluna].map(lambda valor: f"{valor:.4f}")
        texto += ["### Teste final", "", tabela_markdown(tab_teste), ""]
        pares = intervalos[(intervalos.Cenario == cenario) & (intervalos.Modelo_B != "")]
        for linha in pares.itertuples():
            leitura = "inclui zero: diferenca inconclusiva" if linha.IC95_inferior <= 0 <= linha.IC95_superior else "nao inclui zero neste IC pontual"
            texto.append(f"- {linha.Modelo_A} - {linha.Modelo_B}, {linha.Metrica}: "
                         f"{linha.Estimativa:.4f}, IC95% [{linha.IC95_inferior:.4f}, {linha.IC95_superior:.4f}]; {leitura}.")
        texto.append("")
        ncols = 3
        fig, eixos = plt.subplots(int(np.ceil(len(tc) / ncols)), ncols,
                                  figsize=(12, 3.5 * np.ceil(len(tc) / ncols)), squeeze=False)
        for eixo, linha in zip(eixos.flat, tc.itertuples()):
            matriz = np.array([[linha.TN, linha.FP], [linha.FN, linha.TP]])
            ConfusionMatrixDisplay(matriz, display_labels=[0, 1]).plot(ax=eixo, colorbar=False, values_format="d")
            eixo.set_title(linha.Modelo, fontsize=9)
        for eixo in list(eixos.flat)[len(tc):]:
            eixo.axis("off")
        fig.suptitle(f"Teste: {cenario}" + (" [EXECUCAO RAPIDA]" if smoke else ""))
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        fig.savefig(figuras / f"confusao_{cenario}.png", dpi=140)
        plt.close(fig)
        fig, eixo = plt.subplots(figsize=(10, 6))
        posicoes = np.arange(len(rc))
        eixo.barh(posicoes, rc.F1_mean, xerr=rc.F1_std, capsize=3, color="#237d75")
        eixo.set_yticks(posicoes, rc.Modelo, fontsize=8)
        eixo.set(xlabel="F1 medio +/- desvio padrao", title=f"CV: {cenario}" + (" [EXECUCAO RAPIDA]" if smoke else ""))
        fig.tight_layout()
        fig.savefig(figuras / f"f1_cv_{cenario}.png", dpi=160)
        plt.close(fig)

    historico = pd.read_csv(pasta / "historico_treino.csv")
    for cenario in teste.Cenario.unique():
        fig, eixo = plt.subplots(figsize=(9, 5))
        for linha in teste[(teste.Cenario == cenario) & teste.Modelo.isin(PRINCIPAIS)].itertuples():
            h = historico[(historico.Cenario == cenario) & (historico.Etapa == "teste")
                          & (historico.Configuracao == linha.Configuracao)]
            if not h.empty:
                eixo.plot(h.Epoca, h.Perda_validacao, label=f"{linha.Modelo}: validacao")
                eixo.axvline(linha.Melhor_epoca, linestyle=":", alpha=0.35)
        eixo.set(xlabel="Epoca", ylabel="Meio erro quadratico medio",
                 title=f"Parada antecipada: {cenario}" + (" [EXECUCAO RAPIDA]" if smoke else ""))
        eixo.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(figuras / f"parada_antecipada_{cenario}.png", dpi=160)
        plt.close(fig)
    (pasta / "relatorio.md").write_text("\n".join(texto), encoding="utf-8")
