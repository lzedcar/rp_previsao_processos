# === IMPORTAÇÃO DE BIBLIOTECAS ===
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# === CARREGAMENTO DOS DADOS ===
df = pd.read_csv("../dados/tjsp_processos_tratado.csv", sep=";")

# === VISÃO GERAL DOS DADOS ===
print(df.info())
print(df.describe())
print("\nValores ausentes antes do tratamento:")
print(df.isnull().sum())

# === TRATAMENTO DE DADOS ===
df = df.dropna(subset=['TPSent_12_meses_num', 'TPCPL_Dec_2024_num'])
df[['Municipio', 'UF']] = df[['Municipio', 'UF']].fillna('Não informado')
df[['%CP', '%Sus']] = df[['%CP', '%Sus']].fillna('0%')

print("\nValores ausentes após o tratamento:")
print(df.isnull().sum())

# === ANÁLISE DESCRITIVA ===
print("\nResumo estatístico das variáveis numéricas:")
print(df.describe())

# === CORRELAÇÃO COM TEMPO DE TRAMITAÇÃO ===
df_num = df.select_dtypes(include=['float64', 'int64'])
correlacoes = df_num.corr()['TPSent_12_meses_num'].sort_values(ascending=False)

print("\n10 variáveis mais correlacionadas (positivamente) com o tempo de tramitação:")
print(correlacoes.head(10))

print("\n10 variáveis mais correlacionadas (negativamente):")
print(correlacoes.tail(10))

# === VISUALIZAÇÕES ===

# Histograma do tempo de tramitação
plt.figure(figsize=(10, 5))
sns.histplot(df['TPSent_12_meses_num'], bins=30, kde=True)
plt.title('Distribuição do Tempo de Tramitação')
plt.xlabel('Meses até Sentença')
plt.ylabel('Frequência')
plt.tight_layout()
plt.show()

# Mapa de calor das correlações
plt.figure(figsize=(10, 8))
sns.heatmap(df_num.corr(), cmap='coolwarm', annot=False)
plt.title('Mapa de Calor das Correlações')
plt.tight_layout()
plt.show()

# Boxplot por Grau
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x='Grau', y='TPSent_12_meses_num')
plt.title('Tempo de Tramitação por Grau de Jurisdição')
plt.xlabel('Grau')
plt.ylabel('Meses até Sentença')
plt.tight_layout()
plt.show()

# 10 órgãos mais lentos
ranking_lento = df.groupby('Nome orgao')['TPSent_12_meses_num'].mean().sort_values(ascending=False).head(10)
plt.figure(figsize=(10, 6))
sns.barplot(x=ranking_lento.values, y=ranking_lento.index)
plt.title('10 Órgãos com Maior Tempo Médio de Tramitação')
plt.xlabel('Tempo Médio (meses)')
plt.ylabel('Órgão Judicial')
plt.tight_layout()
plt.show()

# Correlação TPCPL x TPSent
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x='TPCPL_Dec_2024_num', y='TPSent_12_meses_num')
plt.title('Correlação entre TPCPL e Tempo de Tramitação')
plt.xlabel('Tempo para Conclusão após Última Sentença (meses)')
plt.ylabel('Tempo de Tramitação (meses)')
plt.tight_layout()
plt.show()

# === ANÁLISE DESCRITIVA NO FORMATO TEXTO ===
"""
Com base na análise exploratória realizada, observou-se que o tempo de tramitação dos processos judiciais
(representado pela variável TPSent_12_meses_num) apresenta forte correlação positiva com o tempo para conclusão
do processo após a última sentença (TPCPL_Dec_2024_num), com coeficiente de 0,50.

Além disso, a variável Conc100_Dec_2024, que representa a quantidade de processos com mais de 100 dias pendentes
de conclusão, também apresentou correlação positiva (0,14), ainda que mais discreta.

Por outro lado, variáveis como SentSM_12_meses e Codigo orgao mostraram correlação negativa com o tempo de tramitação,
sugerindo que órgãos com maior número de sentenças monocráticas e determinadas estruturas organizacionais podem estar
associados a maior agilidade processual.

Essas evidências iniciais apontam para a importância de aspectos tanto operacionais quanto estruturais na duração
dos processos e servem de base para a etapa de modelagem preditiva.
"""

# === Gráfico de dispersão entre TPCPL e Tempo de Tramitação colorido por Grau de Jurisdição ===
plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=df,
    x='TPCPL_Dec_2024_num',
    y='TPSent_12_meses_num',
    hue='Grau',
    palette='Set2',
    alpha=0.7
)
plt.title('Correlação entre TPCPL e Tempo de Tramitação, por Grau')
plt.xlabel('Tempo para Conclusão após Última Sentença (meses)')
plt.ylabel('Tempo de Tramitação até Sentença (meses)')
plt.legend(title='Grau de Jurisdição')
plt.tight_layout()
plt.show()

# Histograma do Índice de Atendimento à Demanda (IAD_12_meses) ===
# Converter a coluna de IAD para numérica, caso ainda esteja como texto com %
df['IAD_12_meses_num'] = df['IAD_12_meses'].str.replace('%', '').str.replace(',', '.').astype(float)

plt.figure(figsize=(10, 6))
sns.histplot(df['IAD_12_meses_num'], bins=30, kde=True, color='skyblue')
plt.title('Distribuição do Índice de Atendimento à Demanda (IAD)')
plt.xlabel('IAD (%)')
plt.ylabel('Frequência')
plt.tight_layout()
plt.show()

# Boxplot do tempo de tramitação nos 10 municípios com mais processos ===
# Selecionar os 10 municípios com mais registros
top_municipios = df['Municipio'].value_counts().head(10).index

plt.figure(figsize=(12, 6))
sns.boxplot(
    data=df[df['Municipio'].isin(top_municipios)],
    x='Municipio',
    y='TPSent_12_meses_num',
    palette='Pastel1'
)
plt.title('Tempo de Tramitação por Município (Top 10 mais frequentes)')
plt.xlabel('Município')
plt.ylabel('Tempo de Tramitação (meses)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Histograma do IAD==
# Converter IAD de string para número
df['IAD_12_meses_num'] = df['IAD_12_meses'].str.replace('%', '').str.replace(',', '.').astype(float)

plt.figure(figsize=(10, 6))
sns.histplot(df['IAD_12_meses_num'], bins=30, kde=True, color='lightcoral')
plt.title('Distribuição do Índice de Atendimento à Demanda (IAD)')
plt.xlabel('IAD (%)')
plt.ylabel('Frequência')
plt.tight_layout()
plt.show()

#Boxplot do tempo de tramitação por municípios mais demandados ==
# Selecionar os 10 municípios com mais registros
top_municipios = df['Municipio'].value_counts().head(10).index

plt.figure(figsize=(12, 6))
sns.boxplot(
    data=df[df['Municipio'].isin(top_municipios)],
    x='Municipio',
    y='TPSent_12_meses_num',
    palette='Pastel2'
)
plt.title('Tempo de Tramitação por Município (Top 10 mais frequentes)')
plt.xlabel('Município')
plt.ylabel('Tempo de Tramitação (meses)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()