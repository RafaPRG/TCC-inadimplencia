# Relatorio de execucao

Resultados do protocolo com selecao interna e teste reservado.

Media +/- desvio padrao: folds externos do desenvolvimento. Cada fold seleciona configuracao/limiar na sua validacao interna. O desvio nao e erro padrao; os treinos se sobrepoem.

AP e average precision, nao a integral trapezoidal da curva PR. Bootstrap: IC95% percentil, estratificado e pareado por registro do teste. Os ICs sao condicionais aos modelos treinados e nao incluem variabilidade de retreinamento; sao pontuais e nao ajustados para comparacoes multiplas.

A base ja foi explorada anteriormente. Estes resultados nao constituem validacao externa inedita. Uma diferenca pequena de F1 nao demonstra superioridade por si so.

## completa: validacao cruzada

| Modelo | F1 | AP | AUC_ROC | Precisao | Recall |
| --- | --- | --- | --- | --- | --- |
| MLP_pesos | 0.8613 +/- 0.0124 | 0.9408 +/- 0.0084 | 0.9924 +/- 0.0012 | 0.9794 +/- 0.0138 | 0.7688 +/- 0.0157 |
| MLP_sem | 0.8619 +/- 0.0076 | 0.9434 +/- 0.0064 | 0.9929 +/- 0.0010 | 0.9819 +/- 0.0107 | 0.7681 +/- 0.0100 |
| MLP_smote_0.3 | 0.8600 +/- 0.0101 | 0.9423 +/- 0.0072 | 0.9927 +/- 0.0011 | 0.9699 +/- 0.0344 | 0.7735 +/- 0.0227 |
| MLP_smote_0.5 | 0.8610 +/- 0.0071 | 0.9435 +/- 0.0064 | 0.9929 +/- 0.0008 | 0.9733 +/- 0.0086 | 0.7721 +/- 0.0149 |
| MLP_smote_1 | 0.8604 +/- 0.0063 | 0.9421 +/- 0.0076 | 0.9927 +/- 0.0011 | 0.9652 +/- 0.0169 | 0.7765 +/- 0.0151 |
| RL_pesos | 0.8625 +/- 0.0073 | 0.9444 +/- 0.0074 | 0.9931 +/- 0.0010 | 0.9736 +/- 0.0191 | 0.7748 +/- 0.0231 |
| RL_sem | 0.8605 +/- 0.0103 | 0.9445 +/- 0.0073 | 0.9931 +/- 0.0011 | 0.9613 +/- 0.0183 | 0.7792 +/- 0.0192 |
| RL_smote_0.3 | 0.8619 +/- 0.0084 | 0.9443 +/- 0.0071 | 0.9929 +/- 0.0010 | 0.9798 +/- 0.0157 | 0.7698 +/- 0.0190 |
| RL_smote_0.5 | 0.8622 +/- 0.0081 | 0.9441 +/- 0.0072 | 0.9928 +/- 0.0010 | 0.9793 +/- 0.0126 | 0.7704 +/- 0.0180 |
| RL_smote_1 | 0.8604 +/- 0.0079 | 0.9437 +/- 0.0072 | 0.9929 +/- 0.0009 | 0.9655 +/- 0.0162 | 0.7762 +/- 0.0171 |
| RandomForest_sem | 0.8646 +/- 0.0103 | 0.9514 +/- 0.0053 | 0.9946 +/- 0.0007 | 0.9609 +/- 0.0406 | 0.7870 +/- 0.0197 |
| Regra_historico_sem | 0.8268 +/- 0.0087 | 0.7288 +/- 0.0116 | 0.8524 +/- 0.0063 | 1.0000 +/- 0.0000 | 0.7049 +/- 0.0126 |

### Teste final

| Modelo | Limiar | F1 | AP | AUC_ROC | Precisao | Recall |
| --- | --- | --- | --- | --- | --- | --- |
| RL_sem | 0.4233 | 0.8639 | 0.9495 | 0.9938 | 0.9527 | 0.7903 |
| RL_smote_1 | 0.9197 | 0.8615 | 0.9501 | 0.9940 | 0.9336 | 0.7997 |
| RL_pesos | 0.9092 | 0.8636 | 0.9498 | 0.9940 | 0.9424 | 0.7970 |
| RL_smote_0.3 | 0.7469 | 0.8641 | 0.9501 | 0.9939 | 0.9512 | 0.7916 |
| RL_smote_0.5 | 0.8319 | 0.8619 | 0.9499 | 0.9939 | 0.9365 | 0.7984 |
| MLP_sem | 0.3721 | 0.8598 | 0.9478 | 0.9935 | 0.9225 | 0.8051 |
| MLP_smote_1 | 0.8457 | 0.8599 | 0.9494 | 0.9939 | 0.9449 | 0.7889 |
| MLP_pesos | 0.7314 | 0.8582 | 0.9417 | 0.9923 | 0.9389 | 0.7903 |
| MLP_smote_0.3 | 0.6927 | 0.8695 | 0.9486 | 0.9937 | 0.9684 | 0.7889 |
| MLP_smote_0.5 | 0.7333 | 0.8563 | 0.9489 | 0.9938 | 0.9180 | 0.8024 |
| Regra_historico_sem | 0.5000 | 0.8381 | 0.7439 | 0.8606 | 1.0000 | 0.7212 |
| RandomForest_sem | 0.3350 | 0.8679 | 0.9587 | 0.9955 | 0.9133 | 0.8268 |

- MLP_sem - RL_sem, F1: -0.0041, IC95% [-0.0119, 0.0041]; inclui zero: diferenca inconclusiva.
- MLP_sem - RL_sem, AUC_ROC: -0.0004, IC95% [-0.0006, -0.0001]; nao inclui zero neste IC pontual.
- MLP_sem - RL_sem, AP: -0.0016, IC95% [-0.0030, -0.0003]; nao inclui zero neste IC pontual.
- MLP_smote_1 - MLP_sem, F1: 0.0001, IC95% [-0.0089, 0.0085]; inclui zero: diferenca inconclusiva.
- MLP_smote_1 - MLP_sem, AUC_ROC: 0.0004, IC95% [0.0001, 0.0008]; nao inclui zero neste IC pontual.
- MLP_smote_1 - MLP_sem, AP: 0.0016, IC95% [-0.0005, 0.0038]; inclui zero: diferenca inconclusiva.
- RL_smote_1 - RL_sem, F1: -0.0024, IC95% [-0.0094, 0.0049]; inclui zero: diferenca inconclusiva.
- RL_smote_1 - RL_sem, AUC_ROC: 0.0002, IC95% [-0.0000, 0.0004]; inclui zero: diferenca inconclusiva.
- RL_smote_1 - RL_sem, AP: 0.0006, IC95% [-0.0007, 0.0020]; inclui zero: diferenca inconclusiva.
- MLP_smote_1 - RL_smote_1, F1: -0.0016, IC95% [-0.0088, 0.0054]; inclui zero: diferenca inconclusiva.
- MLP_smote_1 - RL_smote_1, AUC_ROC: -0.0002, IC95% [-0.0004, 0.0001]; inclui zero: diferenca inconclusiva.
- MLP_smote_1 - RL_smote_1, AP: -0.0007, IC95% [-0.0023, 0.0010]; inclui zero: diferenca inconclusiva.
