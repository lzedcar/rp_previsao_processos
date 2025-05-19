# === revisao_modelagem_aprimorada_final.py ===
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from category_encoders import TargetEncoder

# === CARREGAR DADOS ===
df = pd.read_csv("../dados/tjsp_processos_tratado.csv", sep=";")

# === DEFINIR VARIÁVEIS ===
variaveis_numericas = [
    'TPCPL_Dec_2024_num', 'Conc100_Dec_2024', 'CN_12_meses', 'Desp_12_meses',
    'Sus_Dec_2023', 'Sus_Dec_2024', 'Tbaix_12_meses', 'CP_Dec_2024',
    'SentCM_12_meses', 'SentSM_12_meses'
]
variaveis_categoricas = ['Nome orgao', 'Municipio', 'Grau']

# === REMOVER REGISTROS INCOMPLETOS ===
df = df.dropna(subset=['TPSent_12_meses_num'] + variaveis_numericas + variaveis_categoricas)

# === PREPARAR X E y ===
X_num = df[variaveis_numericas]
X_cat = df[variaveis_categoricas]
y = df['TPSent_12_meses_num']

# === DIVIDIR TREINO E TESTE ===
X_num_train, X_num_test, X_cat_train, X_cat_test, y_train, y_test = train_test_split(
    X_num, X_cat, y, test_size=0.2, random_state=42
)

# === APLICAR TARGET ENCODER APENAS NO TREINO ===
encoder = TargetEncoder()
X_cat_train_enc = encoder.fit_transform(X_cat_train, y_train)
X_cat_test_enc = encoder.transform(X_cat_test)

# === COMBINAR NUMÉRICAS + CATEGÓRICAS ===
X_train_final = pd.concat([X_num_train.reset_index(drop=True), X_cat_train_enc.reset_index(drop=True)], axis=1)
X_test_final = pd.concat([X_num_test.reset_index(drop=True), X_cat_test_enc.reset_index(drop=True)], axis=1)

# === AJUSTAR MODELO RANDOM FOREST ===
modelo = RandomForestRegressor(random_state=42)
modelo.fit(X_train_final, y_train)
y_pred = modelo.predict(X_test_final)

# === MÉTRICAS DE DESEMPENHO ===
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred) ** 0.5
r2 = r2_score(y_test, y_pred)

print("\n=== RESULTADOS DO MODELO FINAL CORRIGIDO ===")
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R²: {r2:.2f}")

# === IMPORTÂNCIA DAS VARIÁVEIS ===
importancias = modelo.feature_importances_
nomes_variaveis = X_train_final.columns

importancia_df = pd.DataFrame({
    'Variável': nomes_variaveis,
    'Importância': importancias
}).sort_values(by='Importância', ascending=False)

# === GRÁFICO ===
plt.figure(figsize=(10, 6))
sns.barplot(data=importancia_df, x='Importância', y='Variável')
plt.title("Importância das Variáveis - Modelo Final Corrigido")
plt.tight_layout()
plt.show()

