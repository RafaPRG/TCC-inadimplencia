# Relatorio de execucao

Resultados do protocolo com selecao interna e teste reservado.

Media +/- desvio padrao: folds externos do desenvolvimento. Cada fold seleciona configuracao/limiar na sua validacao interna. O desvio nao e erro padrao; os treinos se sobrepoem.

AP e average precision, nao a integral trapezoidal da curva PR. Bootstrap: IC95% percentil, estratificado e pareado por registro do teste. Os ICs sao condicionais aos modelos treinados e nao incluem variabilidade de retreinamento; sao pontuais e nao ajustados para comparacoes multiplas.

A base ja foi explorada anteriormente. Estes resultados nao constituem validacao externa inedita. Uma diferenca pequena de F1 nao demonstra superioridade por si so.

## sem_credit_score: validacao cruzada

| Modelo | F1 | AP | AUC_ROC | Precisao | Recall |
| --- | --- | --- | --- | --- | --- |
| MLP_pesos | 0.8203 +/- 0.0106 | 0.8342 +/- 0.0072 | 0.9593 +/- 0.0031 | 0.9906 +/- 0.0135 | 0.7001 +/- 0.0140 |
| MLP_sem | 0.8252 +/- 0.0094 | 0.8363 +/- 0.0072 | 0.9587 +/- 0.0033 | 0.9971 +/- 0.0064 | 0.7038 +/- 0.0119 |
| MLP_smote_0.3 | 0.8254 +/- 0.0085 | 0.8362 +/- 0.0089 | 0.9590 +/- 0.0036 | 1.0000 +/- 0.0000 | 0.7028 +/- 0.0123 |
| MLP_smote_0.5 | 0.8259 +/- 0.0083 | 0.8340 +/- 0.0075 | 0.9577 +/- 0.0038 | 0.9981 +/- 0.0020 | 0.7045 +/- 0.0129 |
| MLP_smote_1 | 0.8237 +/- 0.0095 | 0.8349 +/- 0.0078 | 0.9580 +/- 0.0033 | 0.9914 +/- 0.0044 | 0.7045 +/- 0.0131 |
| RL_pesos | 0.8257 +/- 0.0084 | 0.8359 +/- 0.0071 | 0.9598 +/- 0.0031 | 1.0000 +/- 0.0000 | 0.7032 +/- 0.0122 |
| RL_sem | 0.8254 +/- 0.0079 | 0.8363 +/- 0.0073 | 0.9598 +/- 0.0033 | 1.0000 +/- 0.0000 | 0.7028 +/- 0.0115 |
| RL_smote_0.3 | 0.8248 +/- 0.0089 | 0.8362 +/- 0.0085 | 0.9597 +/- 0.0036 | 0.9981 +/- 0.0043 | 0.7028 +/- 0.0118 |
| RL_smote_0.5 | 0.8257 +/- 0.0071 | 0.8355 +/- 0.0083 | 0.9594 +/- 0.0035 | 1.0000 +/- 0.0000 | 0.7032 +/- 0.0103 |
| RL_smote_1 | 0.8259 +/- 0.0075 | 0.8355 +/- 0.0079 | 0.9593 +/- 0.0033 | 1.0000 +/- 0.0000 | 0.7035 +/- 0.0109 |
| RandomForest_sem | 0.8248 +/- 0.0103 | 0.8386 +/- 0.0091 | 0.9624 +/- 0.0036 | 0.9991 +/- 0.0021 | 0.7025 +/- 0.0158 |
| Regra_historico_sem | 0.8268 +/- 0.0087 | 0.7288 +/- 0.0116 | 0.8524 +/- 0.0063 | 1.0000 +/- 0.0000 | 0.7049 +/- 0.0126 |

### Teste final

| Modelo | Limiar | F1 | AP | AUC_ROC | Precisao | Recall |
| --- | --- | --- | --- | --- | --- | --- |
| RL_sem | 0.4896 | 0.8381 | 0.8483 | 0.9620 | 1.0000 | 0.7212 |
| RL_smote_1 | 0.9837 | 0.8344 | 0.8480 | 0.9619 | 1.0000 | 0.7158 |
| RL_pesos | 0.8951 | 0.8371 | 0.8477 | 0.9621 | 1.0000 | 0.7199 |
| RL_smote_0.3 | 0.8804 | 0.8381 | 0.8484 | 0.9616 | 1.0000 | 0.7212 |
| RL_smote_0.5 | 0.9453 | 0.8381 | 0.8481 | 0.9617 | 1.0000 | 0.7212 |
| MLP_sem | 0.2153 | 0.8383 | 0.8434 | 0.9570 | 0.9981 | 0.7226 |
| MLP_smote_1 | 0.7604 | 0.8310 | 0.8432 | 0.9597 | 0.9981 | 0.7118 |
| MLP_pesos | 0.7050 | 0.8263 | 0.8406 | 0.9571 | 0.9981 | 0.7050 |
| MLP_smote_0.3 | 0.4765 | 0.8367 | 0.8448 | 0.9587 | 0.9963 | 0.7212 |
| MLP_smote_0.5 | 0.8607 | 0.8362 | 0.8463 | 0.9582 | 1.0000 | 0.7185 |
| Regra_historico_sem | 0.5000 | 0.8381 | 0.7439 | 0.8606 | 1.0000 | 0.7212 |
| RandomForest_sem | 0.7700 | 0.8371 | 0.8435 | 0.9621 | 1.0000 | 0.7199 |

- MLP_sem - RL_sem, F1: 0.0003, IC95% [-0.0020, 0.0027]; inclui zero: diferenca inconclusiva.
- MLP_sem - RL_sem, AUC_ROC: -0.0050, IC95% [-0.0075, -0.0024]; nao inclui zero neste IC pontual.
- MLP_sem - RL_sem, AP: -0.0049, IC95% [-0.0102, -0.0001]; nao inclui zero neste IC pontual.
- MLP_smote_1 - MLP_sem, F1: -0.0073, IC95% [-0.0132, -0.0024]; nao inclui zero neste IC pontual.
- MLP_smote_1 - MLP_sem, AUC_ROC: 0.0027, IC95% [0.0014, 0.0039]; nao inclui zero neste IC pontual.
- MLP_smote_1 - MLP_sem, AP: -0.0002, IC95% [-0.0027, 0.0024]; inclui zero: diferenca inconclusiva.
- RL_smote_1 - RL_sem, F1: -0.0037, IC95% [-0.0080, -0.0009]; nao inclui zero neste IC pontual.
- RL_smote_1 - RL_sem, AUC_ROC: -0.0001, IC95% [-0.0007, 0.0004]; inclui zero: diferenca inconclusiva.
- RL_smote_1 - RL_sem, AP: -0.0003, IC95% [-0.0018, 0.0013]; inclui zero: diferenca inconclusiva.
- MLP_smote_1 - RL_smote_1, F1: -0.0034, IC95% [-0.0096, 0.0028]; inclui zero: diferenca inconclusiva.
- MLP_smote_1 - RL_smote_1, AUC_ROC: -0.0022, IC95% [-0.0043, -0.0001]; nao inclui zero neste IC pontual.
- MLP_smote_1 - RL_smote_1, AP: -0.0049, IC95% [-0.0093, -0.0003]; nao inclui zero neste IC pontual.
