# === revisao_modelagem_preditiva.py ===
# Nova modelagem com base nas descobertas da análise dos fatores de atraso

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# === CARREGAR DADOS ===
df = pd.read_csv("../dados/tjsp_processos_tratado.csv", sep=";")

# === SELEÇÃO DAS VARIÁVEIS NUMÉRICAS CONFIÁVEIS ===
variaveis_numericas = [
    'TPCPL_Dec_2024_num',
    'Conc100_Dec_2024',
    'CN_12_meses',
    'Desp_12_meses',
    'Sus_Dec_2023',
    'Sus_Dec_2024',
    'Tbaix_12_meses',
    'CP_Dec_2024',
    'SentCM_12_meses',
    'SentSM_12_meses'
]

df_modelo = df.dropna(subset=['TPSent_12_meses_num'] + variaveis_numericas)

X = df_modelo[variaveis_numericas]
y = df_modelo['TPSent_12_meses_num']

# === TREINO E TESTE ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === MODELAGEM ===
modelos = {
    'Regressão Linear': LinearRegression(),
    'Árvore de Decisão': DecisionTreeRegressor(random_state=42),
    'Random Forest': RandomForestRegressor(random_state=42)
}

resultados = []

for nome, modelo in modelos.items():
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred) ** 0.5
    r2 = r2_score(y_test, y_pred)

    resultados.append({
        'Modelo': nome,
        'MAE': mae,
        'RMSE': rmse,
        'R²': r2
    })

# === EXIBIR RESULTADOS ===
resultados_df = pd.DataFrame(resultados)
print("\nResultados da Modelagem (Revisada):")
print(resultados_df.sort_values(by='R²', ascending=False))

# === GRÁFICO COMPARATIVO DE DESEMPENHO ===
plt.figure(figsize=(10, 5))
sns.barplot(data=resultados_df, x='Modelo', y='R²', palette='Blues_d')
plt.title("Comparativo de R² entre Modelos (Revisão)")
plt.ylabel("Coeficiente de Determinação (R²)")
plt.xlabel("Modelo")
plt.ylim(0, 1)
plt.tight_layout()
plt.show()


