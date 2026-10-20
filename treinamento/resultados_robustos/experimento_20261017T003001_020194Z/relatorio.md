# Relatorio de execucao

Resultados do protocolo com selecao interna e teste reservado.

Media +/- desvio padrao: folds externos do desenvolvimento. Cada fold seleciona configuracao/limiar na sua validacao interna. O desvio nao e erro padrao; os treinos se sobrepoem.

AP e average precision, nao a integral trapezoidal da curva PR. Bootstrap: IC95% percentil, estratificado e pareado por registro do teste. Os ICs sao condicionais aos modelos treinados e nao incluem variabilidade de retreinamento; sao pontuais e nao ajustados para comparacoes multiplas.

A base ja foi explorada anteriormente. Estes resultados nao constituem validacao externa inedita. Uma diferenca pequena de F1 nao demonstra superioridade por si so.

## sem_historico_e_pontuacao: validacao cruzada

| Modelo | F1 | AP | AUC_ROC | Precisao | Recall |
| --- | --- | --- | --- | --- | --- |
| MLP_pesos | 0.3736 +/- 0.0113 | 0.2737 +/- 0.0138 | 0.8649 +/- 0.0071 | 0.2429 +/- 0.0088 | 0.8117 +/- 0.0541 |
| MLP_sem | 0.3707 +/- 0.0089 | 0.2798 +/- 0.0096 | 0.8651 +/- 0.0054 | 0.2558 +/- 0.0143 | 0.6852 +/- 0.0888 |
| MLP_smote_0.3 | 0.3716 +/- 0.0121 | 0.2765 +/- 0.0105 | 0.8658 +/- 0.0059 | 0.2478 +/- 0.0079 | 0.7438 +/- 0.0369 |
| MLP_smote_0.5 | 0.3770 +/- 0.0078 | 0.2772 +/- 0.0092 | 0.8668 +/- 0.0054 | 0.2530 +/- 0.0144 | 0.7498 +/- 0.0706 |
| MLP_smote_1 | 0.3743 +/- 0.0100 | 0.2726 +/- 0.0115 | 0.8650 +/- 0.0054 | 0.2405 +/- 0.0079 | 0.8455 +/- 0.0453 |
| RL_pesos | 0.3720 +/- 0.0060 | 0.2732 +/- 0.0114 | 0.8650 +/- 0.0047 | 0.2401 +/- 0.0056 | 0.8300 +/- 0.0681 |
| RL_sem | 0.3748 +/- 0.0101 | 0.2802 +/- 0.0122 | 0.8668 +/- 0.0052 | 0.2462 +/- 0.0082 | 0.7877 +/- 0.0494 |
| RL_smote_0.3 | 0.3770 +/- 0.0088 | 0.2774 +/- 0.0121 | 0.8663 +/- 0.0050 | 0.2453 +/- 0.0073 | 0.8161 +/- 0.0332 |
| RL_smote_0.5 | 0.3741 +/- 0.0077 | 0.2755 +/- 0.0121 | 0.8657 +/- 0.0047 | 0.2439 +/- 0.0071 | 0.8049 +/- 0.0517 |
| RL_smote_1 | 0.3733 +/- 0.0082 | 0.2722 +/- 0.0115 | 0.8649 +/- 0.0046 | 0.2390 +/- 0.0066 | 0.8543 +/- 0.0517 |
| RandomForest_sem | 0.3909 +/- 0.0060 | 0.2997 +/- 0.0130 | 0.8804 +/- 0.0027 | 0.2704 +/- 0.0069 | 0.7089 +/- 0.0551 |

### Teste final

| Modelo | Limiar | F1 | AP | AUC_ROC | Precisao | Recall |
| --- | --- | --- | --- | --- | --- | --- |
| RL_sem | 0.1109 | 0.3801 | 0.2927 | 0.8690 | 0.2491 | 0.8024 |
| RL_smote_1 | 0.5544 | 0.3738 | 0.2826 | 0.8671 | 0.2436 | 0.8024 |
| RL_pesos | 0.5580 | 0.3765 | 0.2852 | 0.8679 | 0.2440 | 0.8241 |
| RL_smote_0.3 | 0.2886 | 0.3773 | 0.2882 | 0.8680 | 0.2469 | 0.7997 |
| RL_smote_0.5 | 0.3968 | 0.3768 | 0.2867 | 0.8679 | 0.2464 | 0.8011 |
| MLP_sem | 0.1626 | 0.3780 | 0.2949 | 0.8673 | 0.2579 | 0.7077 |
| MLP_smote_1 | 0.5070 | 0.3820 | 0.2830 | 0.8699 | 0.2555 | 0.7564 |
| MLP_pesos | 0.6316 | 0.3629 | 0.2609 | 0.8596 | 0.2464 | 0.6888 |
| MLP_smote_0.3 | 0.2337 | 0.3735 | 0.2844 | 0.8674 | 0.2443 | 0.7930 |
| MLP_smote_0.5 | 0.4080 | 0.3724 | 0.2873 | 0.8689 | 0.2541 | 0.6969 |
| RandomForest_sem | 0.2000 | 0.3791 | 0.3059 | 0.8778 | 0.2670 | 0.6536 |

- MLP_sem - RL_sem, F1: -0.0021, IC95% [-0.0133, 0.0088]; inclui zero: diferenca inconclusiva.
- MLP_sem - RL_sem, AUC_ROC: -0.0017, IC95% [-0.0040, 0.0008]; inclui zero: diferenca inconclusiva.
- MLP_sem - RL_sem, AP: 0.0022, IC95% [-0.0066, 0.0108]; inclui zero: diferenca inconclusiva.
- MLP_smote_1 - MLP_sem, F1: 0.0039, IC95% [-0.0072, 0.0158]; inclui zero: diferenca inconclusiva.
- MLP_smote_1 - MLP_sem, AUC_ROC: 0.0025, IC95% [-0.0007, 0.0054]; inclui zero: diferenca inconclusiva.
- MLP_smote_1 - MLP_sem, AP: -0.0119, IC95% [-0.0231, -0.0006]; nao inclui zero neste IC pontual.
- RL_smote_1 - RL_sem, F1: -0.0063, IC95% [-0.0120, -0.0006]; nao inclui zero neste IC pontual.
- RL_smote_1 - RL_sem, AUC_ROC: -0.0019, IC95% [-0.0032, -0.0006]; nao inclui zero neste IC pontual.
- RL_smote_1 - RL_sem, AP: -0.0100, IC95% [-0.0148, -0.0049]; nao inclui zero neste IC pontual.
- MLP_smote_1 - RL_smote_1, F1: 0.0082, IC95% [-0.0006, 0.0162]; inclui zero: diferenca inconclusiva.
- MLP_smote_1 - RL_smote_1, AUC_ROC: 0.0027, IC95% [0.0003, 0.0051]; nao inclui zero neste IC pontual.
- MLP_smote_1 - RL_smote_1, AP: 0.0003, IC95% [-0.0095, 0.0094]; inclui zero: diferenca inconclusiva.
