import pandas as pd


def preparar_sem_ajuste(df_creditos_ae):
    """Regras fixas aplicadas antes das divisoes (nao estimam nada da base)."""
    df_creditos_ae = df_creditos_ae.drop(columns=["customer_id", "name"])

    # Unico registro com genero XNA.
    df_creditos_ae = df_creditos_ae[df_creditos_ae["gender"] != "XNA"].copy()

    # Valores acima de 300 mil dias sao tratados como ausencia de emprego.
    df_creditos_ae["is_unemployed"] = (
        df_creditos_ae["no_of_days_employed"] > 300000
    ).astype(int)
    df_creditos_ae.loc[
        df_creditos_ae["no_of_days_employed"] > 300000,
        "no_of_days_employed",
    ] = 0

    return df_creditos_ae
