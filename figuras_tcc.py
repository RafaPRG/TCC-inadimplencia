"""Gera as Figuras 4.1 a 4.6 do TCC a partir dos resultados salvos, sem novo treinamento.

Uso: python figuras_tcc.py
     python figuras_tcc.py --resultados PASTA1 PASTA2 PASTA3 PASTA4
"""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import precision_recall_curve, roc_curve

RAIZ = Path(__file__).resolve().parent
CENARIOS = ['completa', 'sem_historico', 'sem_credit_score', 'sem_historico_e_pontuacao']
MODELOS = ['MLP_sem', 'RL_sem', 'MLP_smote_1', 'RL_smote_1']
ROTULOS = ['MLP sem SMOTE', 'RL sem SMOTE', 'MLP + SMOTE 1:1', 'RL + SMOTE 1:1']
CORES = ['#2f6f9f', '#d9853b', '#4d8b57', '#9b5278']
ARQUIVOS = ['resumo_cv.csv', 'teste_final.csv', 'previsoes.csv',
            'intervalos_bootstrap.csv', 'historico_treino.csv']


def caminho_relativo(p):
    p = Path(p).resolve()
    return p.relative_to(RAIZ).as_posix() if p.is_relative_to(RAIZ) else p.name


def sha256(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def carregar(pastas):
    dados, origens, assinaturas = {}, {}, []
    for pasta in pastas:
        m = json.loads((pasta / 'manifesto.json').read_text(encoding='utf-8'))
        if m['Status'] != 'concluido' or m['Somente_smoke']:
            raise ValueError(f'Execucao incompleta ou rapida: {pasta}')
        a = m['Argumentos']
        assinaturas.append((m['Base_SHA256'], a['seed'], a['folds'], a['epocas'],
                            a['paciencia'], a['bootstrap'], a['perfil'], a['extras'],
                            sha256(pasta / 'divisoes.csv')))
        tabelas = {n: pd.read_csv(pasta / n, keep_default_na=False,
                                 float_precision='round_trip') for n in ARQUIVOS}
        for c in a['cenarios']:
            if c not in CENARIOS:
                continue
            if c in dados:
                raise ValueError(f'Cenario repetido: {c}; use --resultados para escolher as pastas.')
            dados[c] = {n: df[df.Cenario == c].copy() for n, df in tabelas.items()}
            origens[c] = {'pasta': caminho_relativo(pasta),
                          'sha256': {n: sha256(pasta / n) for n in ARQUIVOS + ['manifesto.json', 'divisoes.csv']}}
    if set(dados) != set(CENARIOS):
        raise ValueError('Sao necessarios os quatro cenarios completos do TCC.')
    if len(set(assinaturas)) != 1:
        raise ValueError('As execucoes diferem em dados, parametros ou divisoes; nao combinar.')
    if not assinaturas[0][7]:
        raise ValueError('Use uma execucao feita com --extras.')
    return dados, origens


def estilo(ax, grade='both'):
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(axis=grade, alpha=.18)


def salvar(fig, saida, numero, dados):
    nome = f'figura_4_{numero}'
    fig.savefig(saida / f'{nome}.png', dpi=180)
    fig.savefig(saida / f'{nome}.pdf')
    plt.close(fig)
    dados.to_csv(saida / f'{nome}_dados.csv', index=False, encoding='utf-8-sig')


def gerar(d, saida):
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 16,
                         'axes.titlesize': 16, 'axes.labelsize': 16,
                         'legend.fontsize': 13, 'lines.linewidth': 2,
                         'axes.linewidth': 1, 'figure.facecolor': 'white',
                         'pdf.fonttype': 42, 'axes.unicode_minus': True})
    completa = d['completa']
    cv = completa['resumo_cv.csv'].set_index('Modelo').loc[MODELOS]
    teste = completa['teste_final.csv'].set_index('Modelo')
    pred = completa['previsoes.csv'].query("Etapa == 'teste'")
    pares = completa['intervalos_bootstrap.csv']

    # Figura 4.1: media e desvio padrao do F1 nas particoes externas.
    fig, ax = plt.subplots(figsize=(12, 20/3))
    fig.subplots_adjust(left=.086, right=.98, bottom=.133, top=.918)
    x = np.arange(4)
    ax.bar(x, cv.F1_mean, yerr=cv.F1_std, color=CORES, width=.6, capsize=4)
    ax.set_xticks(x, ['MLP\nsem SMOTE', 'RL\nsem SMOTE', 'MLP\nSMOTE 1:1', 'RL\nSMOTE 1:1'])
    ax.set(ylim=(0, 1), ylabel='F1-Score', title='Base completa: média e desvio padrão em cinco folds')
    estilo(ax, 'y')
    for i, r in enumerate(cv.itertuples()):
        ax.text(i, r.F1_mean + r.F1_std + .018,
                f'{r.F1_mean:.4f}\n± {r.F1_std:.4f}'.replace('.', ','), ha='center', va='bottom')
    salvar(fig, saida, 1, cv.reset_index()[['Modelo', 'F1_mean', 'F1_std', 'F1_count']])

    # Figura 4.2: curvas ROC e precisao-recall no teste.
    fig, axs = plt.subplots(1, 2, figsize=(12, 20/3))
    fig.subplots_adjust(left=.086, right=.98, bottom=.29, top=.918, wspace=.25)
    pontos = []
    for m, label, color in zip(MODELOS, ROTULOS, CORES):
        g = pred[pred.Modelo == m]
        fpr, tpr, _ = roc_curve(g.y, g.Probabilidade)
        precision, recall, _ = precision_recall_curve(g.y, g.Probabilidade)
        axs[0].plot(fpr, tpr, color=color, label=label)
        axs[1].step(recall, precision, where='post', color=color, label=label)
        pontos += [{'Modelo': m, 'Curva': 'ROC', 'x': a, 'y': b} for a, b in zip(fpr, tpr)]
        pontos += [{'Modelo': m, 'Curva': 'PR', 'x': a, 'y': b} for a, b in zip(recall, precision)]
    axs[0].plot([0, 1], [0, 1], '--', color='gray', linewidth=1.5)
    axs[1].axhline(pred.drop_duplicates('Registro').y.mean(), linestyle=':', color='gray', linewidth=2)
    axs[0].set(title='ROC', xlabel='Taxa de falsos positivos', ylabel='Recall')
    axs[1].set(title='Precisão-recall', xlabel='Recall', ylabel='Precisão')
    for ax in axs:
        ax.set(xlim=(-.02, 1.02), ylim=(-.02, 1.02)); estilo(ax)
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=2, bbox_to_anchor=(.5, .012))
    salvar(fig, saida, 2, pd.DataFrame(pontos))

    # Figura 4.3: diferencas de F1 com intervalos de 95% do bootstrap.
    comparacoes = [('MLP_sem', 'RL_sem'), ('MLP_smote_1', 'MLP_sem'),
                   ('RL_smote_1', 'RL_sem'), ('MLP_smote_1', 'RL_smote_1')]
    linhas = [pares[(pares.Modelo_A == a) & (pares.Modelo_B == b) & (pares.Metrica == 'F1')].iloc[0] for a, b in comparacoes]
    ic = pd.DataFrame(linhas)
    fig, ax = plt.subplots(figsize=(12, 6.6))
    fig.subplots_adjust(left=.192, right=.98, bottom=.14, top=.918)
    ax.errorbar(ic.Estimativa, np.arange(4),
                xerr=np.vstack([ic.Estimativa - ic.IC95_inferior, ic.IC95_superior - ic.Estimativa]),
                fmt='o', color=CORES[0], capsize=6, markersize=8)
    ax.axvline(0, linestyle='--', color='gray')
    ax.set_yticks(np.arange(4), ['MLP − RL\n(sem SMOTE)', 'MLP: com − sem\n(SMOTE 1:1)',
                                'RL: com − sem\n(SMOTE 1:1)', 'MLP − RL\n(com SMOTE 1:1)'])
    ax.invert_yaxis()
    ax.set_xlim(-.015, .012)
    ax.set_xticks([-.015, -.010, -.005, 0, .005, .010])
    ax.set(xlabel='Diferença de F1 (primeiro modelo menos segundo)',
           title='Teste da base completa: IC95% bootstrap pareado')
    estilo(ax, 'x')
    salvar(fig, saida, 3, ic)

    # Figura 4.4: F1 e precisao media nos quatro cenarios.
    fig, axs = plt.subplots(1, 2, figsize=(12, 20/3))
    fig.subplots_adjust(left=.086, right=.978, bottom=.26, top=.918, wspace=.23)
    ab = []
    for j, m in enumerate(MODELOS[:2]):
        linhas = []
        for c in CENARIOS:
            rc = d[c]['resumo_cv.csv'].set_index('Modelo').loc[m]
            tc = d[c]['teste_final.csv'].set_index('Modelo').loc[m]
            linhas.append({'Cenario': c, 'Modelo': m, 'F1_mean': rc.F1_mean, 'F1_std': rc.F1_std, 'AP': tc.AP})
        ab += linhas
        valores = pd.DataFrame(linhas)
        pos = np.arange(4) + (j - .5) * .32
        axs[0].bar(pos, valores.F1_mean, yerr=valores.F1_std, width=.3, capsize=4, color=CORES[j], label=ROTULOS[j])
        axs[1].bar(pos, valores.AP, width=.3, color=CORES[j])
    prev = d['completa']['teste_final.csv'].iloc[0]
    axs[1].axhline((prev.TP + prev.FN) / prev.N_avaliacao, ls='--', color='gray', lw=1.5,
                   label='Proporção de inadimplentes no teste')
    axs[0].set(title='F1 nas partições (média ± DP)', ylabel='F1')
    axs[1].set(title='Precisão média no teste final', ylabel='Precisão média (AP)')
    for ax in axs:
        ax.set_xticks(np.arange(4), ['Completa', 'Sem\nhistórico', 'Sem\npontuação',
                                     'Sem histórico\ne pontuação'], fontsize=14)
        ax.set_ylim(0, 1); estilo(ax, 'y')
    handles, labels = axs[0].get_legend_handles_labels()
    h2, l2 = axs[1].get_legend_handles_labels()
    fig.legend(handles + h2, labels + l2, ncol=3, loc='lower center', bbox_to_anchor=(.5, .005))
    salvar(fig, saida, 4, pd.DataFrame(ab))

    # Figura 4.5: perda de validacao no ajuste final.
    fig, ax = plt.subplots(figsize=(12, 20/3))
    fig.subplots_adjust(left=.11, right=.98, bottom=.143, top=.918)
    historicos = []
    for idx in [0, 2]:
        m = MODELOS[idx]; r = teste.loc[m]
        h = completa['historico_treino.csv']
        h = h[(h.Etapa == 'teste') & (h.Configuracao == r.Configuracao)].copy()
        if len(h) != r.Epocas or not (h.Epoca == r.Melhor_epoca).any():
            raise ValueError(f'Historico inconsistente: {m}')
        h['Modelo'] = m; h['Melhor_epoca'] = r.Melhor_epoca
        historicos.append(h)
        ax.plot(h.Epoca, h.Perda_validacao, color=CORES[idx], linewidth=2.5,
                label=f'{ROTULOS[idx]} (melhor época: {int(r.Melhor_epoca)})')
        ax.axvline(r.Melhor_epoca, linestyle=':', color=CORES[idx], alpha=.6)
    ax.set(title='Validação do ajuste final: cenário completo', xlabel='Época',
           ylabel='Metade do erro quadrático médio')
    estilo(ax); ax.legend(loc='upper right')
    salvar(fig, saida, 5, pd.concat(historicos, ignore_index=True))

    # Figura 4.6: calibracao em dez grupos por quantis.
    fig, ax = plt.subplots(figsize=(12, 20/3))
    fig.subplots_adjust(left=.086, right=.965, bottom=.14, top=.918)
    calibracao = []
    for m, label, color in zip(MODELOS, ROTULOS, CORES):
        g = pred[pred.Modelo == m]
        obs, prob = calibration_curve(g.y, g.Probabilidade, n_bins=10, strategy='quantile')
        ax.plot(prob, obs, marker='o', color=color, label=label)
        calibracao += [{'Modelo': m, 'Grupo': i+1, 'Probabilidade_media': p, 'Fracao_positivos': o}
                       for i, (p, o) in enumerate(zip(prob, obs))]
    ax.plot([0, 1], [0, 1], '--', color='gray', label='Referência ideal')
    ax.set(xlim=(0, 1), ylim=(0, 1), title='Teste da base completa: curvas de calibração',
           xlabel='Probabilidade prevista média no grupo', ylabel='Fração observada de inadimplentes')
    estilo(ax); ax.legend(loc='upper left')
    salvar(fig, saida, 6, pd.DataFrame(calibracao))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resultados', nargs='+', type=Path)
    parser.add_argument('--saida', type=Path, default=RAIZ / 'figuras_tcc')
    args = parser.parse_args()
    pastas = args.resultados or sorted((RAIZ / 'treinamento/resultados_robustos').glob('experimento_*'))
    dados, origens = carregar(pastas)
    args.saida.mkdir(parents=True, exist_ok=True)
    gerar(dados, args.saida)
    registro = {'script_sha256': sha256(Path(__file__)), 'origens': origens,
                'descricao': 'Figuras geradas a partir dos resultados salvos, sem novo treinamento.',
                'arquivos_sha256': {p.name: sha256(p) for p in sorted(args.saida.glob('figura_4_*')) if p.is_file()}}
    (args.saida / 'proveniencia.json').write_text(json.dumps(registro, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Seis figuras (PNG/PDF), dados (CSV) e proveniencia: {args.saida}')


if __name__ == '__main__':
    main()
