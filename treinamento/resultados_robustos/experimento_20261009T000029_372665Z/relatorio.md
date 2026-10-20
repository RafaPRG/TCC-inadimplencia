# Relatorio de execucao

Resultados do protocolo com selecao interna e teste reservado.

Media +/- desvio padrao: folds externos do desenvolvimento. Cada fold seleciona configuracao/limiar na sua validacao interna. O desvio nao e erro padrao; os treinos se sobrepoem.

AP e average precision, nao a integral trapezoidal da curva PR. Bootstrap: IC95% percentil, estratificado e pareado por registro do teste. Os ICs sao condicionais aos modelos treinados e nao incluem variabilidade de retreinamento; sao pontuais e nao ajustados para comparacoes multiplas.

A base ja foi explorada anteriormente. Estes resultados nao constituem validacao externa inedita. Uma diferenca pequena de F1 nao demonstra superioridade por si so.

## sem_historico: validacao cruzada

| Modelo | F1 | AP | AUC_ROC | Precisao | Recall |
| --- | --- | --- | --- | --- | --- |
| MLP_pesos | 0.8267 +/- 0.0173 | 0.9298 +/- 0.0076 | 0.9917 +/- 0.0009 | 0.8719 +/- 0.0390 | 0.7870 +/- 0.0189 |
| MLP_sem | 0.8456 +/- 0.0135 | 0.9383 +/- 0.0069 | 0.9923 +/- 0.0008 | 0.9103 +/- 0.0311 | 0.7904 +/- 0.0251 |
| MLP_smote_0.3 | 0.8315 +/- 0.0146 | 0.9364 +/- 0.0098 | 0.9924 +/- 0.0011 | 0.9068 +/- 0.0475 | 0.7691 +/- 0.0167 |
| MLP_smote_0.5 | 0.8321 +/- 0.0167 | 0.9352 +/- 0.0088 | 0.9922 +/- 0.0010 | 0.8968 +/- 0.0435 | 0.7779 +/- 0.0308 |
| MLP_smote_1 | 0.8382 +/- 0.0132 | 0.9349 +/- 0.0083 | 0.9922 +/- 0.0010 | 0.8984 +/- 0.0172 | 0.7860 +/- 0.0227 |
| RL_pesos | 0.8340 +/- 0.0162 | 0.9363 +/- 0.0085 | 0.9924 +/- 0.0009 | 0.8857 +/- 0.0368 | 0.7890 +/- 0.0244 |
| RL_sem | 0.8445 +/- 0.0152 | 0.9385 +/- 0.0079 | 0.9925 +/- 0.0008 | 0.8925 +/- 0.0099 | 0.8019 +/- 0.0279 |
| RL_smote_0.3 | 0.8423 +/- 0.0147 | 0.9382 +/- 0.0083 | 0.9923 +/- 0.0009 | 0.8974 +/- 0.0234 | 0.7944 +/- 0.0291 |
| RL_smote_0.5 | 0.8433 +/- 0.0168 | 0.9378 +/- 0.0084 | 0.9923 +/- 0.0009 | 0.8993 +/- 0.0239 | 0.7941 +/- 0.0193 |
| RL_smote_1 | 0.8410 +/- 0.0166 | 0.9376 +/- 0.0083 | 0.9923 +/- 0.0009 | 0.8934 +/- 0.0265 | 0.7951 +/- 0.0276 |
| RandomForest_sem | 0.8593 +/- 0.0088 | 0.9514 +/- 0.0063 | 0.9945 +/- 0.0008 | 0.9476 +/- 0.0384 | 0.7873 +/- 0.0230 |

### Teste final

| Modelo | Limiar | F1 | AP | AUC_ROC | Precisao | Recall |
| --- | --- | --- | --- | --- | --- | --- |
| RL_sem | 0.4976 | 0.8472 | 0.9439 | 0.9936 | 0.9112 | 0.7916 |
| RL_smote_1 | 0.9178 | 0.8474 | 0.9443 | 0.9936 | 0.9152 | 0.7889 |
| RL_pesos | 0.9286 | 0.8407 | 0.9426 | 0.9935 | 0.9238 | 0.7713 |
| RL_smote_0.3 | 0.7475 | 0.8477 | 0.9444 | 0.9936 | 0.9087 | 0.7943 |
| RL_smote_0.5 | 0.8440 | 0.8492 | 0.9441 | 0.9936 | 0.9196 | 0.7889 |
| MLP_sem | 0.4554 | 0.8434 | 0.9406 | 0.9930 | 0.8807 | 0.8092 |
| MLP_smote_1 | 0.8333 | 0.8307 | 0.9387 | 0.9932 | 0.9153 | 0.7605 |
| MLP_pesos | 0.8243 | 0.8332 | 0.9340 | 0.9926 | 0.8845 | 0.7876 |
| MLP_smote_0.3 | 0.5243 | 0.8490 | 0.9444 | 0.9936 | 0.8962 | 0.8065 |
| MLP_smote_0.5 | 0.7470 | 0.8505 | 0.9430 | 0.9934 | 0.9118 | 0.7970 |
| RandomForest_sem | 0.4550 | 0.8636 | 0.9588 | 0.9956 | 0.9745 | 0.7754 |

- MLP_sem - RL_sem, F1: -0.0038, IC95% [-0.0138, 0.0064]; inclui zero: diferenca inconclusiva.
- MLP_sem - RL_sem, AUC_ROC: -0.0006, IC95% [-0.0010, -0.0003]; nao inclui zero neste IC pontual.
- MLP_sem - RL_sem, AP: -0.0033, IC95% [-0.0056, -0.0010]; nao inclui zero neste IC pontual.
- MLP_smote_1 - MLP_sem, F1: -0.0127, IC95% [-0.0272, 0.0017]; inclui zero: diferenca inconclusiva.
- MLP_smote_1 - MLP_sem, AUC_ROC: 0.0002, IC95% [-0.0003, 0.0008]; inclui zero: diferenca inconclusiva.
- MLP_smote_1 - MLP_sem, AP: -0.0019, IC95% [-0.0057, 0.0019]; inclui zero: diferenca inconclusiva.
- RL_smote_1 - RL_sem, F1: 0.0002, IC95% [-0.0065, 0.0069]; inclui zero: diferenca inconclusiva.
- RL_smote_1 - RL_sem, AUC_ROC: 0.0000, IC95% [-0.0001, 0.0002]; inclui zero: diferenca inconclusiva.
- RL_smote_1 - RL_sem, AP: 0.0004, IC95% [-0.0006, 0.0015]; inclui zero: diferenca inconclusiva.
- MLP_smote_1 - RL_smote_1, F1: -0.0166, IC95% [-0.0268, -0.0073]; nao inclui zero neste IC pontual.
- MLP_smote_1 - RL_smote_1, AUC_ROC: -0.0004, IC95% [-0.0007, -0.0001]; nao inclui zero neste IC pontual.
- MLP_smote_1 - RL_smote_1, AP: -0.0056, IC95% [-0.0081, -0.0032]; nao inclui zero neste IC pontual.
