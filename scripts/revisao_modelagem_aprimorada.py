# === revisao_modelagem_aprimorada.py ===
# Revisão da modelagem com categóricas reintroduzidas de forma controlada

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from category_encoders import TargetEncoder

# === CARREGAR OS DADOS ===
df = pd.read_csv("../dados/tjsp_processos_tratado.csv", sep=";")

# === REMOVER NULOS NAS VARIÁVEIS USADAS ===
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
variaveis_categoricas = ['Nome orgao', 'Municipio', 'Grau']

df = df.dropna(subset=['TPSent_12_meses_num'] + variaveis_numericas + variaveis_categoricas)

# === FILTRAR CATEGÓRICAS COM POUCA FREQUÊNCIA ===
# Mínimo de 5 processos por categoria
for var in ['Nome orgao', 'Municipio']:
    freq = df[var].value_counts()
    categorias_validas = freq[freq >= 3].index

# === ENCODER PARA VARIÁVEIS CATEGÓRICAS ===
encoder = TargetEncoder(cols=variaveis_categoricas)
df_encoded = encoder.fit_transform(df[variaveis_categoricas], df['TPSent_12_meses_num'])

# === COMBINAR VARIÁVEIS ===
X = pd.concat([df[variaveis_numericas].reset_index(drop=True), df_encoded.reset_index(drop=True)], axis=1)
y = df['TPSent_12_meses_num'].reset_index(drop=True)

# === DIVIDIR TREINO E TESTE ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === AJUSTAR RANDOM FOREST ===
modelo = RandomForestRegressor(random_state=42)
modelo.fit(X_train, y_train)
y_pred = modelo.predict(X_test)

# === AVALIAR DESEMPENHO ===
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred) ** 0.5
r2 = r2_score(y_test, y_pred)

print("\n=== RESULTADOS DO MODELO REVISADO (APIMORADO) ===")
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R²: {r2:.2f}")

# === IMPORTÂNCIA DAS VARIÁVEIS ===
importancias = modelo.feature_importances_
nomes_variaveis = X.columns

importancia_df = pd.DataFrame({
    'Variável': nomes_variaveis,
    'Importância': importancias
}).sort_values(by='Importância', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(data=importancia_df, x='Importância', y='Variável')
plt.title("Importância das Variáveis - Modelo Aprimorado (Revisado)")
plt.tight_layout()
plt.show()

