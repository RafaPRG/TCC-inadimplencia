import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from numpy.testing import assert_allclose, assert_array_equal
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.neural_network import MLPClassifier

from treinamento.avaliacao import bootstrap_pareado, selecionar_limiar, selecionar_limiar_custo
from treinamento.experimento import (Configuracao, dividir_desenvolvimento_teste,
                                     executar_divisao, planos_validacao, treinar_candidato)
from treinamento.mlp import PMC1, g
from treinamento.preprocessamento import aplicar_smote_fold, criar_preprocessador
from tratamento.tratamento_bd_ae import preparar_sem_ajuste


class TestMLP(unittest.TestCase):
    def test_gradiente_numerico_incluindo_pesos_e_bias(self):
        for peso in [1.0, 3.7]:
            rede = PMC1(3, 4, 2, random_state=12)
            x, y = np.array([0.2, -0.3, 0.7]), np.array([1.0, 0.0])
            rede.propagar(x)
            gradientes = rede.gradientes(y, peso)
            for matriz, analitico in zip((rede.W1, rede.W2), gradientes):
                numerico = np.empty_like(matriz)
                for indice in np.ndindex(matriz.shape):
                    original = matriz[indice]
                    matriz[indice] = original + 1e-6
                    mais = 0.5 * peso * np.sum((y - rede.propagar(x)) ** 2)
                    matriz[indice] = original - 1e-6
                    menos = 0.5 * peso * np.sum((y - rede.propagar(x)) ** 2)
                    matriz[indice] = original
                    numerico[indice] = (mais - menos) / 2e-6
                assert_allclose(analitico, numerico, atol=1e-9, rtol=1e-5)

    def test_embaralha_cada_epoca_com_reprodutibilidade(self):
        exemplos = np.column_stack([np.arange(12) / 12, np.arange(12) % 2])
        original = exemplos.copy()
        ordens = []
        redes = []
        for _ in range(2):
            rede = PMC1(1, 3, 1, random_state=4)
            ordem = []
            propagar = rede.propagar

            def observar(x):
                ordem.append(float(x[0]))
                return propagar(x)

            rede.propagar = observar
            rede.treinar(exemplos, max_epocas=3, epsilon=0)
            redes.append(rede)
            ordens.append(np.array(ordem).reshape(3, 12))
        assert_array_equal(ordens[0], ordens[1])
        self.assertFalse(np.array_equal(ordens[0][0], ordens[0][1]))
        for epoca in ordens[0]:
            assert_array_equal(np.sort(epoca), exemplos[:, 0])
        assert_allclose(redes[0].W1, redes[1].W1)
        assert_array_equal(exemplos, original)

    def test_parada_restaura_melhores_pesos(self):
        rede = PMC1(1, 3, 1)
        exemplos = np.array([[0.0, 0.0], [1.0, 1.0]])
        pesos = []

        def prever_controlado(X):
            pesos.append((rede.W1.copy(), rede.W2.copy()))
            return np.array([[0.0], [1.0]]) if len(pesos) == 1 else np.array([[1.0], [0.0]])

        rede.prever_probabilidades = prever_controlado
        epocas, _ = rede.treinar(exemplos, validacao=exemplos, paciencia=2, max_epocas=10)
        self.assertEqual(epocas, 3)
        self.assertEqual(rede.melhor_epoca_, 1)
        assert_allclose(rede.W1, pesos[0][0])
        assert_allclose(rede.W2, pesos[0][1])

    def test_forward_equivalente_sklearn_mesmos_pesos(self):
        X = np.random.default_rng(2).random((20, 3))
        rede = PMC1(3, 4, 1)
        sklearn = MLPClassifier(hidden_layer_sizes=(4,), activation="logistic", random_state=42)
        sklearn.partial_fit(X, np.arange(20) % 2, classes=[0, 1])
        sklearn.coefs_ = [rede.W1[:, 1:].T.copy(), rede.W2[:, 1:].T.copy()]
        sklearn.intercepts_ = [-rede.W1[:, 0].copy(), -rede.W2[:, 0].copy()]
        assert_allclose(rede.prever_probabilidades(X)[:, 0], sklearn.predict_proba(X)[:, 1], atol=1e-14, rtol=0)
        assert_allclose(rede.prever_probabilidades(X), [rede.propagar(x) for x in X])

    def test_sanidade_aprendizado_mesma_topologia(self):
        rng = np.random.default_rng(1)
        y = np.tile([0, 1], 100)
        X = y[:, None] * np.ones((1, 2)) + rng.normal(0, 0.05, (200, 2))
        modelo, _, _ = treinar_candidato(Configuracao("MLP", "sem", 4, 0.2),
                                          X[:150], y[:150], X[150:175], y[150:175], 42, 40, 5)
        p = modelo.prever_probabilidades(X[175:])[:, 0]
        self.assertGreater(roc_auc_score(y[175:], p), 0.95)
        self.assertGreater(f1_score(y[175:], p >= 0.5), 0.9)

    def test_sigmoide_estavel(self):
        with np.errstate(over="raise"):
            assert_allclose(g(np.array([-1e6, 0, 1e6])), [0, 0.5, 1], atol=1e-300)


class TestProtocolo(unittest.TestCase):
    def test_preprocessamento_nao_aprende_validacao(self):
        treino = pd.DataFrame({"age": [1, np.nan, 3], "occupation_type": ["A", "A", "B"]})
        validacao = pd.DataFrame({"age": [100, np.nan], "occupation_type": ["Z", "Z"]})
        prep = criar_preprocessador(treino)
        prep.fit(treino)
        resultado = prep.transform(validacao)
        self.assertEqual(resultado[0, 0], 49.5)
        self.assertEqual(resultado[1, 0], 0.5)
        assert_allclose(prep.named_transformers_["numericas"]["imputacao"].statistics_, [2])
        self.assertNotIn("Z", prep.named_transformers_["categoricas"]["codificacao"].categories_[0])

    def test_regras_fixas_preservam_nulos(self):
        df = pd.DataFrame({"customer_id": [1, 2], "name": ["A", "B"], "gender": ["F", "XNA"],
                           "no_of_days_employed": [400000, 10], "credit_score": [np.nan, 500]})
        saida = preparar_sem_ajuste(df)
        self.assertEqual(len(saida), 1)
        self.assertTrue(saida.credit_score.isna().all())
        self.assertEqual(saida.no_of_days_employed.iloc[0], 0)
        self.assertEqual(saida.is_unemployed.iloc[0], 1)
        self.assertEqual(df.no_of_days_employed.iloc[0], 400000)

    def test_splits_disjuntos_e_avaliacao_externa_unica(self):
        y = np.tile([0] * 9 + [1], 100)
        dev, teste = dividir_desenvolvimento_teste(y, 42)
        externos = []
        for _, treino, validacao, avaliacao in planos_validacao(y, dev, 5, 42):
            grupos = list(map(set, [treino, validacao, avaliacao, teste]))
            for i, a in enumerate(grupos):
                for b in grupos[i + 1:]:
                    self.assertFalse(a & b)
            externos.extend(avaliacao)
        assert_array_equal(np.sort(externos), np.sort(dev))

    def test_smote_proporcao_e_reprodutibilidade(self):
        X = np.arange(44, dtype=float).reshape(22, 2)
        y = np.array([0] * 20 + [1] * 2)
        xb, yb = aplicar_smote_fold(X, y, sampling_strategy=0.5)
        self.assertEqual(np.sum(yb == 1), 10)
        assert_array_equal(xb, aplicar_smote_fold(X, y, sampling_strategy=0.5)[0])
        assert_array_equal(X, np.arange(44).reshape(22, 2))

    def test_limiar_exato_acima_de_095_e_empates(self):
        y = np.array([0, 0, 1, 1])
        p = np.array([0.96, 0.97, 0.98, 0.99])
        limiar, f1 = selecionar_limiar(y, p)
        self.assertEqual(limiar, 0.98)
        self.assertEqual(f1, 1)
        for probabilidades in [p, np.array([0.3, 0.3, 0.6, 0.6]), np.ones(4)]:
            limiar, f1 = selecionar_limiar(y, probabilidades)
            melhor = max(f1_score(y, probabilidades >= l) for l in np.unique(probabilidades))
            self.assertEqual(f1_score(y, probabilidades >= limiar), melhor)
            self.assertAlmostEqual(f1, melhor)

    def test_limiar_custo_contra_busca_exaustiva(self):
        y, p = np.array([0, 1, 0, 1]), np.array([0.3, 0.3, 0.6, 0.8])
        for custo in [1, 2, 5]:
            limiar = selecionar_limiar_custo(y, p, custo)

            def perda(l):
                pred = p >= l
                return np.sum(pred & (y == 0)) + custo * np.sum(~pred & (y == 1))

            self.assertEqual(perda(limiar), min(perda(l) for l in [0.3, 0.6, 0.8, 1.1]))

    def test_bootstrap_pareia_registros(self):
        linhas = pd.DataFrame({"Cenario": "completa", "Modelo": "RL_sem", "Registro": range(10),
                               "y": [0, 1] * 5, "Probabilidade": np.linspace(0, 1, 10), "Limiar": 0.5})
        copia = linhas.copy().iloc[::-1]
        copia["Modelo"] = "MLP_sem"
        resultado = bootstrap_pareado(pd.concat([linhas, copia]), repeticoes=10)
        pares = resultado[resultado.Modelo_B != ""]
        assert_allclose(pares[["Estimativa", "IC95_inferior", "IC95_superior"]], 0)

    def test_alvo_externo_nao_muda_selecao_ou_predicoes(self):
        rng = np.random.default_rng(2)
        X = pd.DataFrame({"age": rng.random(100), "credit_score": rng.random(100)})
        y = np.tile([0, 1], 50)
        args = SimpleNamespace(seed=42, epocas=2, paciencia=2, extras=False)
        configs = [Configuracao("MLP", "sem", 3, 0.01), Configuracao("MLP", "sem", 3, 0.1),
                   Configuracao("RL")]
        contexto = {"Cenario": "completa", "Etapa": "cv", "Fold": 1}
        with tempfile.TemporaryDirectory() as temporario:
            saidas = []
            for alvo in [y, np.r_[y[:80], 1 - y[80:]]]:
                saidas.append(executar_divisao(X, alvo, np.arange(100), np.arange(60), np.arange(60, 80),
                                               np.arange(80, 100), configs, contexto, args, Path(temporario)))
        for antes, depois in zip(saidas[0][0], saidas[1][0]):
            self.assertEqual(antes["Configuracao"], depois["Configuracao"])
            self.assertEqual(antes["Limiar"], depois["Limiar"])
        for antes, depois in zip(saidas[0][3], saidas[1][3]):
            assert_allclose(antes.Probabilidade, depois.Probabilidade)


if __name__ == "__main__":
    unittest.main()
