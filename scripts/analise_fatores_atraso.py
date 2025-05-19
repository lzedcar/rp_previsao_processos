# === analise_fatores_atraso.py ===
# Análise complementar dos fatores que influenciam o tempo de tramitação dos processos

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# === CARREGAR OS DADOS ===
df = pd.read_csv(r"C:\Users\Windows\Arquivos de sistema\Desktop\tcc_previsao_processos\dados\tjsp_processos_tratado.csv", sep=";")

# === REMOVER REGISTROS INCOMPLETOS RELEVANTES ===
df = df.dropna(subset=['TPSent_12_meses_num', 'Nome orgao', 'Municipio', 'Grau'])

# === TEMPO MÉDIO POR ÓRGÃO ===
tempo_por_orgao = df.groupby('Nome orgao')['TPSent_12_meses_num'].mean().sort_values(ascending=False)
print("\nTempo médio por órgão:")
print(tempo_por_orgao.head(10))

# === TEMPO MÉDIO POR MUNICÍPIO ===
tempo_por_municipio = df.groupby('Municipio')['TPSent_12_meses_num'].mean().sort_values(ascending=False)
print("\nTempo médio por município:")
print(tempo_por_municipio.head(10))

# === TEMPO MÉDIO POR GRAU ===
tempo_por_grau = df.groupby('Grau')['TPSent_12_meses_num'].mean()
print("\nTempo médio por grau:")
print(tempo_por_grau)

# === VOLUME DE PROCESSOS POR ÓRGÃO ===
volume_por_orgao = df['Nome orgao'].value_counts()
print("\nVolume de processos por órgão:")
print(volume_por_orgao.head(10))

# === VOLUME DE PROCESSOS POR MUNICÍPIO ===
volume_por_municipio = df['Municipio'].value_counts()
print("\nVolume de processos por município:")
print(volume_por_municipio.head(10))

# === TABELA CRUZADA: TEMPO MÉDIO + QUANTIDADE DE PROCESSOS POR ÓRGÃO ===
resumo_orgao = df.groupby('Nome orgao').agg({
    'TPSent_12_meses_num': 'mean',
    'Nome orgao': 'count'
}).rename(columns={
    'TPSent_12_meses_num': 'Tempo_medio_meses',
    'Nome orgao': 'Qtd_processos'
}).sort_values(by='Tempo_medio_meses', ascending=False)

print("\nResumo por órgão (tempo médio e volume):")
print(resumo_orgao.head(10))

# === GRÁFICOS ===

# Gráfico 1: Top 10 órgãos com maior tempo médio
plt.figure(figsize=(10, 5))
tempo_por_orgao.head(10).plot(kind='barh', color='salmon')
plt.title("Top 10 órgãos com maior tempo médio de tramitação")
plt.xlabel("Tempo médio (meses)")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# Gráfico 2: Top 10 órgãos com maior volume de processos
plt.figure(figsize=(10, 5))
volume_por_orgao.head(10).plot(kind='barh', color='skyblue')
plt.title("Top 10 órgãos com maior volume de processos")
plt.xlabel("Quantidade de processos")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# Gráfico 3: Comparação de tempo médio por grau
plt.figure(figsize=(6, 4))
sns.barplot(x=tempo_por_grau.index, y=tempo_por_grau.values)
plt.title("Tempo médio de tramitação por Grau")
plt.ylabel("Tempo médio (meses)")
plt.xlabel("Grau")
plt.tight_layout()
plt.show()

# === ANÁLISE REFINADA: Filtrar órgãos com pelo menos 5 processos ===
print("\n=== ANÁLISE REFINADA ===")
print("Aplicando filtro: considerar apenas órgãos com 5 ou mais processos.")

# Contar número de processos por órgão
volume_orgao = df['Nome orgao'].value_counts()
orgaos_validos = volume_orgao[volume_orgao >= 5].index

# Criar novo DataFrame filtrado
df_filtrado = df[df['Nome orgao'].isin(orgaos_validos)]

# Recalcular tempo médio por órgão (após filtro)
tempo_filtrado = df_filtrado.groupby('Nome orgao')['TPSent_12_meses_num'].mean().sort_values(ascending=False)
print("\nTempo médio por órgão (≥ 5 processos):")
print(tempo_filtrado.head(10))

# Recalcular volume de processos por órgão (após filtro)
volume_filtrado = df_filtrado['Nome orgao'].value_counts()
print("\nVolume de processos por órgão (≥ 5):")
print(volume_filtrado.head(10))

# Tabela cruzada resumo por órgão (após filtro)
resumo_filtrado = df_filtrado.groupby('Nome orgao').agg({
    'TPSent_12_meses_num': 'mean',
    'Nome orgao': 'count'
}).rename(columns={
    'TPSent_12_meses_num': 'Tempo_medio_meses',
    'Nome orgao': 'Qtd_processos'
}).sort_values(by='Tempo_medio_meses', ascending=False)

print("\nResumo por órgão (tempo médio e volume - ≥ 5 processos):")
print(resumo_filtrado.head(10))

# Gráfico: Top 10 órgãos com maior tempo médio (filtrados)
plt.figure(figsize=(10, 5))
tempo_filtrado.head(10).plot(kind='barh', color='darkorange')
plt.title("Top 10 órgãos com maior tempo médio (≥ 5 processos)")
plt.xlabel("Tempo médio (meses)")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# Gráfico: Top 10 órgãos com maior volume (filtrados)
plt.figure(figsize=(10, 5))
volume_filtrado.head(10).plot(kind='barh', color='steelblue')
plt.title("Top 10 órgãos com maior volume de processos (≥ 5)")
plt.xlabel("Quantidade de processos")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()
