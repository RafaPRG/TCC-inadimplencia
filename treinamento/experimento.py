"""Experimento completo: CSV bruto, selecao interna, validacao cruzada e teste final."""

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import sys
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.utils.class_weight import compute_sample_weight
from threadpoolctl import threadpool_limits

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tratamento.tratamento_bd_ae import preparar_sem_ajuste
from treinamento.avaliacao import calcular_metricas, selecionar_limiar, selecionar_limiar_custo
from treinamento.mlp import PMC1
from treinamento.preprocessamento import aplicar_smote_fold, criar_preprocessador
from treinamento.relatorios import gerar_relatorios


RAIZ = Path(__file__).resolve().parents[1]
COLUNA_ALVO = "credit_card_default"
CENARIOS = {
    "completa": [],
    "sem_historico": ["prev_defaults", "default_in_last_6months"],
    "sem_credit_score": ["credit_score"],
    "sem_historico_e_pontuacao": ["prev_defaults", "default_in_last_6months", "credit_score"],
}


@dataclass(frozen=True)
class Configuracao:
    familia: str
    estrategia: str = "sem"
    neuronios: int = 0
    eta: float = 0.0

    @property
    def grupo(self):
        return f"{self.familia}_{self.estrategia}"

    @property
    def identificador(self):
        return f"{self.grupo}_h{self.neuronios}_eta{self.eta:g}"


def configuracoes(perfil="completo", rapido=False, extras=False):
    estrategias = ["sem", "smote_1"]
    if perfil == "completo":
        estrategias += ["pesos", "smote_0.3", "smote_0.5"]
    grade = [(5, 0.01), (50, 0.01), (50, 0.05), (50, 0.1)]
    if rapido:
        grade = [(5, 0.01), (5, 0.1)]
    configs = [Configuracao("RL", e) for e in estrategias]
    configs += [Configuracao("MLP", e, h, eta) for e in estrategias for h, eta in grade]
    configs += [Configuracao("Regra_historico")]
    if extras:
        configs.append(Configuracao("RandomForest"))
    return configs


def carregar_base(caminho):
    bruto = pd.read_csv(caminho)
    if COLUNA_ALVO not in bruto or bruto[COLUNA_ALVO].isna().any():
        raise ValueError("Alvo ausente ou incompleto")
    if set(bruto[COLUNA_ALVO].unique()) != {0, 1}:
        raise ValueError("O alvo deve conter exatamente as classes 0 e 1")
    if ("customer_id" not in bruto or bruto.customer_id.isna().any()
            or bruto.customer_id.duplicated().any()):
        raise ValueError("Identificadores ausentes/repetidos: revisar split por cliente")
    dados = preparar_sem_ajuste(bruto)
    auditoria = {"Registros_brutos": len(bruto), "Registros_validos": len(dados),
                 "Removidos_gender_XNA": int((bruto.gender == "XNA").sum()),
                 "Nulos_antes_imputacao": dados.isna().sum().to_dict(),
                 "Classes": dados[COLUNA_ALVO].value_counts().to_dict()}
    return dados, auditoria


def dividir_desenvolvimento_teste(y, seed):
    return train_test_split(np.arange(len(y)), test_size=0.2, stratify=y, random_state=seed)


def planos_validacao(y, desenvolvimento, folds, seed):
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    planos = []
    for fold, (treino, avaliacao) in enumerate(cv.split(desenvolvimento, y[desenvolvimento]), 1):
        internos = desenvolvimento[treino]
        ajuste, validacao = train_test_split(internos, test_size=0.2,
                                           stratify=y[internos], random_state=seed + fold)
        planos.append((fold, ajuste, validacao, desenvolvimento[avaliacao]))
    return planos


def treinar_candidato(config, X, y, X_val, y_val, seed, epocas, paciencia):
    inicio = time.perf_counter()
    pesos = compute_sample_weight("balanced", y) if config.estrategia == "pesos" else None
    if config.familia == "MLP":
        modelo = PMC1(X.shape[1], config.neuronios, 1, random_state=seed)
        n, erros = modelo.treinar(np.column_stack([X, y]), eta=config.eta,
                                  max_epocas=epocas, validacao=np.column_stack([X_val, y_val]),
                                  paciencia=paciencia, pesos_amostras=pesos)
        historico = [{"Epoca": i + 1, "Perda_treino": float(e),
                      "Perda_validacao": modelo.historico_validacao_[i]}
                     for i, e in enumerate(erros)]
        melhor_epoca = modelo.melhor_epoca_
    else:
        if config.familia == "RL":
            modelo = LogisticRegression(max_iter=2000, random_state=seed,
                                        class_weight="balanced" if pesos is not None else None)
        elif config.familia == "RandomForest":
            modelo = RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=1)
        else:
            raise ValueError(f"Familia desconhecida: {config.familia}")
        modelo.fit(X, y)
        historico, n, melhor_epoca = [], 0, 0
    return modelo, {"Epocas": n, "Melhor_epoca": melhor_epoca,
                    "Tempo_s": time.perf_counter() - inicio}, historico


def prever(modelo, X):
    if isinstance(modelo, PMC1):
        return modelo.prever_probabilidades(X)[:, 0]
    return modelo.predict_proba(X)[:, 1]


def regra_historico(X):
    """Regra fixa: inadimplente se houver inadimplencia anterior ou recente."""
    return ((X["prev_defaults"] > 0) | (X["default_in_last_6months"] > 0)).to_numpy(dtype=float)


def executar_divisao(X, y, registros, ajuste, validacao, avaliacao, configs, contexto,
                     args, pasta, escolhas_fixas=None):
    seed = args.seed + int(contexto["Fold"])
    preprocessador = criar_preprocessador(X.iloc[ajuste])
    Xt = preprocessador.fit_transform(X.iloc[ajuste])
    Xv = preprocessador.transform(X.iloc[validacao])
    yt, yv = y[ajuste], y[validacao]
    cache = {"sem": (Xt, yt), "pesos": (Xt, yt)}
    candidatos, historicos, melhores = [], [], {}
    for config in configs:
        if config.familia == "Regra_historico":
            if not {"prev_defaults", "default_in_last_6months"}.issubset(X.columns):
                continue
            modelo, info, historia = None, {"Epocas": 0, "Melhor_epoca": 0, "Tempo_s": 0.0}, []
            pv = regra_historico(X.iloc[validacao])
            limiar = 0.5
        else:
            if escolhas_fixas is not None and escolhas_fixas.get(config.grupo) != config.identificador:
                continue
            if config.estrategia not in cache:
                proporcao = float(config.estrategia.split("_")[1])
                cache[config.estrategia] = aplicar_smote_fold(Xt, yt, seed, proporcao)
            Xmodelo, ymodelo = cache[config.estrategia]
            modelo, info, historia = treinar_candidato(config, Xmodelo, ymodelo, Xv, yv,
                                                       seed, args.epocas, args.paciencia)
            pv = prever(modelo, Xv)
            limiar, _ = selecionar_limiar(yv, pv)
        metricas = calcular_metricas(yv, pv, limiar)
        candidato = {**contexto, "Modelo": config.grupo, "Configuracao": config.identificador,
                     **asdict(config), "Limiar": limiar, **info, **metricas}
        candidatos.append(candidato)
        historicos.extend({**contexto, "Configuracao": config.identificador, **h} for h in historia)
        anterior = melhores.get(config.grupo)
        criterio = (metricas["F1"], metricas["AP"])
        if anterior is None or criterio > anterior[0]:
            melhores[config.grupo] = (criterio, modelo, candidato, pv)

    # A particao de avaliacao so e usada depois de todas as escolhas internas.
    Xa = preprocessador.transform(X.iloc[avaliacao])
    resultados, previsoes, custos = [], [], []
    for grupo, (_, modelo, candidato, pv) in melhores.items():
        p = regra_historico(X.iloc[avaliacao]) if modelo is None else prever(modelo, Xa)
        limiar = candidato["Limiar"]
        linha = {**contexto, "Modelo": grupo, "Configuracao": candidato["Configuracao"],
                 "Limiar": limiar, "F1_validacao": candidato["F1"],
                 "Melhor_epoca": candidato["Melhor_epoca"], "Epocas": candidato["Epocas"],
                 "N_preditoras": Xt.shape[1], "N_ajuste": len(ajuste),
                 "N_validacao": len(validacao), "N_avaliacao": len(avaliacao),
                 **calcular_metricas(y[avaliacao], p, limiar)}
        resultados.append(linha)
        previsoes.append(pd.DataFrame({**contexto, "Modelo": grupo,
                                      "Registro": registros[avaliacao], "y": y[avaliacao],
                                      "Probabilidade": p, "Limiar": limiar}))
        if contexto["Etapa"] == "teste":
            artefatos = pasta / "modelos"
            artefatos.mkdir(exist_ok=True)
            joblib.dump({"preprocessador": preprocessador, "modelo": modelo,
                         "colunas": list(X.columns), "configuracao": candidato,
                         "regra": "prev_defaults > 0 OR default_in_last_6months > 0" if modelo is None else None},
                        artefatos / f"{contexto['Cenario']}_{grupo}.joblib")
            if args.extras:
                for custo_fn in [1, 2, 5, 10]:
                    lc = selecionar_limiar_custo(yv, pv, custo_fn)
                    mc = calcular_metricas(y[avaliacao], p, lc)
                    custos.append({**contexto, "Modelo": grupo, "Custo_FN": custo_fn,
                                   "Custo_FP": 1, "Limiar": lc,
                                   "Custo_medio": (custo_fn * mc["FN"] + mc["FP"]) / len(avaliacao),
                                   **mc})
    return resultados, candidatos, historicos, previsoes, custos


def caminho_relativo(caminho):
    caminho = Path(caminho).resolve()
    return caminho.relative_to(RAIZ).as_posix() if caminho.is_relative_to(RAIZ) else caminho.name


def hash_arquivo(caminho):
    return hashlib.sha256(Path(caminho).read_bytes()).hexdigest()


def registrar_divisao(linhas, registros, fold, ajuste, validacao, avaliacao, etapa):
    for nome, indices in [("ajuste", ajuste), ("validacao", validacao), ("avaliacao", avaliacao)]:
        linhas.extend({"Etapa": etapa, "Fold": fold, "Papel": nome, "Registro": int(registros[i])}
                      for i in indices)


def rodar_experimentos(args):
    if args.folds < 2 or args.epocas < 1 or args.paciencia < 1 or args.bootstrap < 2:
        raise ValueError("Folds/bootstrap >= 2; epocas/paciencia >= 1")
    if args.rapido:
        args.epocas = min(args.epocas, 2)
        args.bootstrap = min(args.bootstrap, 30)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    pasta = args.saida / (f"smoke_{stamp}" if args.rapido else f"experimento_{stamp}")
    pasta.mkdir(parents=True, exist_ok=False)
    dados, auditoria = carregar_base(args.base)
    if args.rapido:
        indices, _ = train_test_split(np.arange(len(dados)), train_size=min(1200, len(dados) // 2),
                                      stratify=dados[COLUNA_ALVO], random_state=args.seed)
        dados = dados.iloc[np.sort(indices)]
    y = dados[COLUNA_ALVO].to_numpy(dtype=int)
    registros = dados.index.to_numpy()
    X = dados.drop(columns=COLUNA_ALVO)
    desenvolvimento, teste = dividir_desenvolvimento_teste(y, args.seed)
    planos = planos_validacao(y, desenvolvimento, args.folds, args.seed)
    ajuste_final, validacao_final = train_test_split(desenvolvimento, test_size=0.2,
                                                    stratify=y[desenvolvimento], random_state=args.seed)
    divisoes = []
    for fold, ajuste, validacao, avaliacao in planos:
        registrar_divisao(divisoes, registros, fold, ajuste, validacao, avaliacao, "cv")
    registrar_divisao(divisoes, registros, 0, ajuste_final, validacao_final, teste, "teste")
    pd.DataFrame(divisoes).to_csv(pasta / "divisoes.csv", index=False)
    configs = configuracoes(args.perfil, args.rapido, args.extras)
    manifesto = {"Status": "em_execucao", "Inicio_UTC": stamp, "Somente_smoke": args.rapido,
                 "Argumentos": {k: caminho_relativo(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                 "Base_SHA256": hash_arquivo(args.base), "Auditoria": auditoria,
                 "N_experimento": len(y), "N_desenvolvimento": len(desenvolvimento), "N_teste": len(teste),
                 "Python": platform.python_version(), "Plataforma": platform.platform(),
                 "Versoes": {p: version(p) for p in ["numpy", "pandas", "scipy", "scikit-learn",
                                                     "imbalanced-learn", "matplotlib", "joblib", "threadpoolctl"]},
                 "Fontes_SHA256": {str(p.relative_to(RAIZ)): hash_arquivo(p)
                                   for pasta_codigo in (RAIZ / "treinamento", RAIZ / "tratamento")
                                   for p in pasta_codigo.glob("*.py")},
                 "Configuracoes": [asdict(c) for c in configs],
                 "Selecao": "F1 validacao interna; desempate AP; limiar exato PR",
                 "Teste_final": "Configuracao por media das validacoes internas; novo ajuste/validacao em desenvolvimento",
                 "Limites": ["Base ja explorada anteriormente; nao constitui validacao externa inedita",
                             "CV externa com holdout interno, nao k-fold interno",
                             "IC bootstrap condicional aos modelos/limiares, pontual, sem ajuste de multiplicidade",
                             "SMOTE comum interpola colunas one-hot; SMOTENC nao implementado",
                             "Regra historico nao executada na ablacao sem_historico"]}
    manifesto_path = pasta / "manifesto.json"
    manifesto_path.write_text(json.dumps(manifesto, indent=2, ensure_ascii=False), encoding="utf-8")
    resultados, candidatos, historicos, previsoes, custos, selecoes = [], [], [], [], [], []

    def acumular(saida):
        for destino, linhas in zip((resultados, candidatos, historicos, previsoes, custos), saida):
            destino.extend(linhas)
        pd.DataFrame(resultados).to_csv(pasta / "metricas.csv", index=False)
        pd.DataFrame(candidatos).to_csv(pasta / "validacao_candidatos.csv", index=False)
        pd.DataFrame(historicos).to_csv(pasta / "historico_treino.csv", index=False)
        pd.concat(previsoes, ignore_index=True).to_csv(pasta / "previsoes.csv", index=False)

    try:
        with threadpool_limits(limits=1):
            for cenario in args.cenarios:
                Xc = X.drop(columns=CENARIOS[cenario])
                for fold, ajuste, validacao, avaliacao in planos:
                    print(f"[{cenario}] fold {fold}/{args.folds}: {len(configs)} candidatos", flush=True)
                    contexto = {"Cenario": cenario, "Etapa": "cv", "Fold": fold}
                    acumular(executar_divisao(Xc, y, registros, ajuste, validacao, avaliacao,
                                              configs, contexto, args, pasta))
                cv_val = pd.DataFrame(candidatos)
                cv_val = cv_val[(cv_val.Cenario == cenario) & (cv_val.Etapa == "cv")]
                medias = cv_val.groupby(["Modelo", "Configuracao"], as_index=False)[["F1", "AP"]].mean()
                escolhidos = medias.sort_values(["F1", "AP", "Configuracao"], ascending=[False, False, True])
                escolhidos = escolhidos.drop_duplicates("Modelo")
                fixas = dict(zip(escolhidos.Modelo, escolhidos.Configuracao))
                selecoes.extend({"Cenario": cenario, **linha} for linha in escolhidos.to_dict("records"))
                pd.DataFrame(selecoes).to_csv(pasta / "configuracoes_finais.csv", index=False)
                print(f"[{cenario}] ajuste final e avaliacao no teste reservado", flush=True)
                acumular(executar_divisao(Xc, y, registros, ajuste_final, validacao_final, teste,
                                          configs, {"Cenario": cenario, "Etapa": "teste", "Fold": 0},
                                          args, pasta, escolhas_fixas=fixas))
            if custos:
                pd.DataFrame(custos).to_csv(pasta / "custos.csv", index=False)
            print("Gerando tabelas, curvas e intervalos bootstrap...", flush=True)
            gerar_relatorios(pasta, args.bootstrap, args.seed, args.rapido, args.extras)
        manifesto["Status"] = "concluido"
    except BaseException as exc:
        manifesto["Status"] = "interrompido_ou_falhou"
        manifesto["Erro"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        manifesto["Fim_UTC"] = datetime.now(timezone.utc).isoformat()
        manifesto_path.write_text(json.dumps(manifesto, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Resultados: {pasta}", flush=True)
    return pasta


def criar_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=RAIZ / "tratamento/database/CreditosAE.csv")
    parser.add_argument("--saida", type=Path, default=RAIZ / "treinamento/resultados_robustos")
    parser.add_argument("--perfil", choices=["minimo", "completo"], default="completo")
    parser.add_argument("--cenarios", nargs="+", choices=list(CENARIOS), default=list(CENARIOS))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epocas", type=int, default=50)
    parser.add_argument("--paciencia", type=int, default=5)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--rapido", action="store_true", help="Execucao rapida de verificacao (subamostra e 2 epocas); nao usar como resultado")
    parser.add_argument("--extras", action="store_true", help="Random Forest, cenarios de custo e calibracao")
    return parser


if __name__ == "__main__":
    rodar_experimentos(criar_parser().parse_args())
