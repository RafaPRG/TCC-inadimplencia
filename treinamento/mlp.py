import numpy as np


def gerar_matriz(j, i, intervalo=(-1, 1), rng=None):
    """Matriz j x i com valores uniformes no intervalo (pesos iniciais)."""
    a, b = intervalo
    rng = np.random.default_rng() if rng is None else rng
    return (b - a) * rng.random((j, i)) + a


def g(x, beta=1):
    """Funcao logistica (sigmoide)."""
    return 1.0 / (1 + np.exp(np.clip(-beta * x, -709, 709)))


def d_g(y):
    """Derivada da sigmoide escrita em funcao da propria saida y = g(x)."""
    return y * (1 - y)


class PMC1:
    """Perceptron multicamadas com uma camada escondida, treinado por
    retropropagacao com gradiente descendente estocastico (um exemplo por vez),
    conforme Silva, Spatti e Flauzino (2016)."""

    def __init__(self, num_entrada, num_escondida, num_saida, random_state=42):
        self.num_c0 = num_entrada
        self.num_c1 = num_escondida
        self.num_c2 = num_saida

        # A coluna extra de cada matriz guarda o peso do bias (entrada -1).
        self.rng = np.random.default_rng(random_state)
        self.W1 = gerar_matriz(self.num_c1, self.num_c0 + 1, rng=self.rng)
        self.W2 = gerar_matriz(self.num_c2, self.num_c1 + 1, rng=self.rng)

        self.propagar(np.ones(self.num_c0))

    def propagar(self, sinais):
        """Fase forward: retorna a saida da rede para um exemplo."""
        if len(sinais) != self.num_c0:
            msg = "Numero de sinais diferente do esperado"
            raise ValueError(msg)

        self.Y0 = np.concatenate(([-1.0], np.asarray(sinais, dtype=float)))

        saida_escondida = g(self.W1 @ self.Y0)
        self.Y1 = np.concatenate(([-1.0], saida_escondida))

        self.Y2 = g(self.W2 @ self.Y1)

        return self.Y2

    def gradientes(self, desejados, peso=1.0):
        """Gradientes da perda 0.5 * peso * soma((D - Y)**2)."""
        D = np.asarray(desejados, dtype=float)
        if D.shape != (self.num_c2,):
            raise ValueError("Numero de desejados diferente do esperado")
        delta2 = peso * (self.Y2 - D) * d_g(self.Y2)
        delta1 = (self.W2[:, 1:].T @ delta2) * d_g(self.Y1[1:])
        return np.outer(delta1, self.Y0), np.outer(delta2, self.Y1)

    def retropropagar(self, desejados, eta, peso=1.0):
        """Fase backward: atualiza W1 e W2 e retorna o erro do exemplo."""
        D = np.array(desejados)
        if len(D) != self.num_c2:
            msg = "Numero de desejados diferente do esperado"
            raise ValueError(msg)

        grad1, grad2 = self.gradientes(D, peso=peso)
        self.W2 -= eta * grad2
        self.W1 -= eta * grad1

        erro = 0.5 * peso * np.sum((D - self.Y2) ** 2)
        return erro

    def treinar(self, exemplos, eta=0.05, epsilon=1e-6, max_epocas=1000,
                validacao=None, paciencia=5, pesos_amostras=None):
        """Treinamento online com nova permutacao a cada epoca.

        Com validacao, aplica parada antecipada (paciencia) e restaura os
        pesos da melhor epoca. Sem validacao, para quando a variacao do erro
        de treino fica abaixo de epsilon.
        """
        exemplos = np.asarray(exemplos, dtype=float)
        if exemplos.ndim != 2 or exemplos.shape[1] != self.num_c0 + self.num_c2:
            raise ValueError("Formato de exemplos invalido")
        if not len(exemplos) or max_epocas < 1 or eta <= 0 or paciencia < 1 or epsilon < 0:
            raise ValueError("Dados ou parametros de treinamento invalidos")
        if not np.isfinite(exemplos).all():
            raise ValueError("Exemplos devem ser finitos")
        pesos = np.ones(len(exemplos)) if pesos_amostras is None else np.asarray(pesos_amostras)
        if pesos.shape != (len(exemplos),) or not np.isfinite(pesos).all() or np.any(pesos <= 0):
            raise ValueError("Pesos devem ser positivos, finitos e alinhados aos exemplos")
        if validacao is not None:
            validacao = np.asarray(validacao, dtype=float)
            if (validacao.ndim != 2 or validacao.shape[1] != exemplos.shape[1]
                    or not len(validacao) or not np.isfinite(validacao).all()):
                raise ValueError("Validacao invalida")
        p = self.num_c2
        feqm = []
        num_epocas, eqm = 0, 0

        self.historico_validacao_ = []
        self.melhor_epoca_ = 0
        melhor_erro, sem_melhora, melhores_pesos = np.inf, 0, None
        while num_epocas < max_epocas:
            eqm_anterior = eqm
            soma_erro = 0.0
            for indice in self.rng.permutation(len(exemplos)):
                um_exemplo = exemplos[indice]
                x, d = um_exemplo[:-p], um_exemplo[-p:]
                self.propagar(x)
                soma_erro += self.retropropagar(d, eta, peso=pesos[indice])
            eqm = soma_erro / float(pesos.sum())
            feqm.append(eqm)
            num_epocas += 1
            delta_eqm = abs(eqm - eqm_anterior)

            if validacao is not None:
                saidas = self.prever_probabilidades(validacao[:, :-p])
                erro_val = float(0.5 * np.mean(np.sum((validacao[:, -p:] - saidas) ** 2, axis=1)))
                self.historico_validacao_.append(erro_val)
                if erro_val < melhor_erro - epsilon:
                    melhor_erro, sem_melhora = erro_val, 0
                    self.melhor_epoca_ = num_epocas
                    melhores_pesos = (self.W1.copy(), self.W2.copy())
                else:
                    sem_melhora += 1
                if sem_melhora >= paciencia:
                    break
            elif delta_eqm <= epsilon:
                break

        if melhores_pesos is not None:
            self.W1, self.W2 = melhores_pesos
        else:
            self.melhor_epoca_ = num_epocas

        return num_epocas, np.array(feqm)

    def prever_probabilidades(self, X):
        """Propagacao vetorizada para varios exemplos (sem atualizar pesos)."""
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != self.num_c0:
            raise ValueError("Formato de entrada invalido")
        oculta = g(X @ self.W1[:, 1:].T - self.W1[:, 0])
        return g(oculta @ self.W2[:, 1:].T - self.W2[:, 0])
