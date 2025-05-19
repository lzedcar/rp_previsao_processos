import pandas as pd
import re

# Leitura da base original
df = pd.read_csv("../dados/tjsp_processos_sp.csv", sep=";")

# Função para converter tempo em meses
def convert_to_months(valor):
    if pd.isna(valor):
        return None
    valor = str(valor).lower()
    anos = meses = 0
    match_ano = re.search(r"(\d+)\s*ano", valor)
    match_mes = re.search(r"(\d+)\s*mes", valor)
    if match_ano:
        anos = int(match_ano.group(1))
    if match_mes:
        meses = int(match_mes.group(1))
    return anos * 12 + meses

# Aplicação da função
df["TPSent_12_meses_num"] = df["TPSent_12_meses"].apply(convert_to_months)
df["TPCPL_Dec_2024_num"] = df["TPCPL_Dec_2024"].apply(convert_to_months)

# Salvar a base tratada
df.to_csv("../dados/tjsp_processos_tratado.csv", sep=";", index=False)
print("Base tratada salva com sucesso!")

# Tratar valores ausentes

# 1. Remover linhas com valores ausentes nas variáveis numéricas essenciais para a modelagem
df = df.dropna(subset=[
    'TPSent_12_meses_num', 
    'TPCPL_Dec_2024_num'
])

# 2. Remover também linhas com IAD_12_meses ausente (fator relevante de atraso)
df = df.dropna(subset=['IAD_12_meses'])

# 3. Preencher colunas textuais com valor padrão
df[['Municipio', 'UF']] = df[['Municipio', 'UF']].fillna('Não informado')

# 4. Substituir valores ausentes em colunas de percentual por "0%" (evita erro na análise)
df[['%CP', '%Sus']] = df[['%CP', '%Sus']].fillna('0%')

# 5. (Recomendo) Substituir campos de texto 'TC_Dec_2024' e 'TPCPL_Dec_2024' ausentes por 'Não informado'
df[['TC_Dec_2024', 'TPCPL_Dec_2024']] = df[['TC_Dec_2024', 'TPCPL_Dec_2024']].fillna('Não informado')

# Verificação final de ausentes
print("\nValores ausentes após o tratamento:")
print(df.isnull().sum())

# Salvar o arquivo tratado
df.to_csv("../dados/tjsp_processos_tratado.csv", sep=";", index=False)
print(f"Arquivo tratado salvo com sucesso com {df.shape[0]} linhas.")
