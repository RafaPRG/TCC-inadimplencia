# Predição de inadimplência de crédito com MLP

Código do Trabalho de Conclusão de Curso "Predição de inadimplência de crédito via redes neurais MLP: um estudo comparativo com Regressão Logística e avaliação do SMOTE" (Bacharelado em Sistemas de Informação, CEFET-MG Campus Varginha).

O trabalho compara uma rede Perceptron Multicamadas (MLP), implementada diretamente em NumPy, com a Regressão Logística na base AmExpert CodeLab 2021, e avalia o efeito do SMOTE e de pesos de classe.

Autor: Rafael Ponce Ribeiro Gouvea
Orientador: Prof. Douglas Machado Tavares
Coorientador: Prof. Nilton César

## Estrutura

```
tratamento/
  tratamento_bd_ae.py      regras fixas aplicadas à base bruta
treinamento/
  mlp.py                   MLP com uma camada escondida (propagação, retropropagação e parada antecipada)
  preprocessamento.py      imputação, escala, one-hot e SMOTE
  avaliacao.py             métricas, escolha de limiar e bootstrap pareado
  experimento.py           protocolo completo de treinamento e avaliação
  relatorios.py            tabelas, figuras e relatório de cada execução
  resultados_robustos/     resultados usados no TCC (um cenário por pasta)
  resultados_sensibilidade_epocas/  análise complementar com até 200 épocas
figuras_tcc.py             gera as Figuras 4.1 a 4.6 do TCC
figuras_tcc/               figuras geradas (PNG, PDF e dados em CSV)
tests/                     testes automatizados
```

## Instalação

Python 3.11.

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Base de dados

A base não está no repositório. Baixe-a no Kaggle:
https://www.kaggle.com/datasets/pradip11/amexpert-codelab-2021/data

Salve o arquivo como `tratamento/database/CreditosAE.csv`. O arquivo usado no trabalho tem 45.528 registros e SHA-256 `8f3c79565e540cf4bfaf24bc09a3a6d379bc945c5072d376df55f53ad64fa106`.

## Execução

Testes:

```powershell
python -m unittest discover -s tests -v
```

Verificação rápida (subamostra e duas épocas, só para checar se tudo roda):

```powershell
python -m treinamento.experimento --rapido --extras
```

Experimento do TCC (quatro cenários, cinco partições, até 50 épocas, 2.000 reamostragens):

```powershell
python -m treinamento.experimento --extras
```

A execução completa pode levar algumas horas, pois a MLP é treinada exemplo a exemplo. Cada execução cria uma pasta `experimento_DATA` com os resultados.

Análise complementar com até 200 épocas (Apêndice C do TCC):

```powershell
python -m treinamento.experimento --cenarios completa --extras --epocas 200 --saida treinamento/resultados_sensibilidade_epocas
```

Figuras do TCC, a partir dos resultados salvos (requer o arquivo `previsoes.csv` de cada execução, gerado ao rodar o experimento):

```powershell
python figuras_tcc.py
```

## Protocolo

1. Remove identificadores e o único registro com `gender = XNA`. Valores de `no_of_days_employed` acima de 300.000 viram 0, com o indicador `is_unemployed`.
2. Reserva 20% da base, com estratificação, para o teste final.
3. Divide o restante em cinco partições externas. Em cada rodada, a parte de treino é separada em 80% para ajuste e 20% para validação interna.
4. Imputação, escala, codificação e SMOTE são ajustados apenas nos dados de ajuste.
5. A validação interna escolhe a configuração, a época de parada da MLP e o limiar (maior F1).
6. Para o teste final, a configuração é escolhida pela média das validações internas, o modelo é ajustado novamente e avaliado uma única vez no teste.
7. Os intervalos de 95% vêm de 2.000 reamostragens bootstrap estratificadas e pareadas no teste.

Modelos: MLP própria, Regressão Logística, floresta aleatória (referência) e uma regra fixa baseada no histórico de inadimplência. Estratégias: sem balanceamento, SMOTE 1:1, 0,5 e 0,3, e pesos de classe.

Cenários: base completa; sem histórico de inadimplência; sem pontuação de crédito; sem histórico e sem pontuação.

## Resultados incluídos

| Pasta | Conteúdo |
| --- | --- |
| `resultados_robustos/experimento_20261009T000028_607629Z` | Base completa |
| `resultados_robustos/experimento_20261009T000029_372665Z` | Sem histórico |
| `resultados_robustos/experimento_20261009T000030_387386Z` | Sem pontuação de crédito |
| `resultados_robustos/experimento_20261017T003001_020194Z` | Sem histórico e sem pontuação |
| `resultados_sensibilidade_epocas/experimento_20261019T182728_227173Z` | Base completa com até 200 épocas |

Cada pasta traz o manifesto da execução (parâmetros, versões e hashes), as métricas por partição e no teste, os intervalos de confiança, o histórico de treino da MLP e as figuras. As previsões individuais, as divisões e os modelos serializados não foram incluídos por causa do tamanho; eles são gerados novamente ao rodar o experimento.

Os manifestos registram os hashes do código no momento de cada execução. Depois disso, o código passou por revisão de comentários e remoção de funções não utilizadas, sem alteração na lógica do experimento.

Na base completa, a MLP sem balanceamento obteve F1 de 0,8598 no teste, e a Regressão Logística, 0,8639. A diferença (−0,0041) teve intervalo de 95% de [−0,0119; 0,0041], sem vantagem conclusiva para nenhum dos modelos.
