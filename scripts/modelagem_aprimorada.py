# === modelagem_aprimorada.py ===
# Objetivo: Aprimorar o modelo de previsão incluindo variáveis categóricas relevantes

# === IMPORTAÇÃO DE BIBLIOTECAS ===
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from category_encoders import TargetEncoder

# === CARREGAMENTO DOS DADOS ===
df = pd.read_csv("../dados/tjsp_processos_tratado.csv", sep=";")

# === LIMPEZA DOS REGISTROS INCOMPLETOS ===
df = df.dropna(subset=[
    'TPSent_12_meses_num', 'TPCPL_Dec_2024_num', 'Conc100_Dec_2024',
    'CN_12_meses', 'Desp_12_meses', 'Sus_Dec_2023', 'Sus_Dec_2024',
    'Tbaix_12_meses', 'CP_Dec_2024', 'SentCM_12_meses', 'SentSM_12_meses',
    'Municipio', 'Nome orgao', 'Grau'
])

# === DEFINIR VARIÁVEL ALVO ===
y = df['TPSent_12_meses_num']

# === VARIÁVEIS NUMÉRICAS ORIGINAIS ===
X_num = df[[
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
]]

# === CODIFICAÇÃO DAS VARIÁVEIS CATEGÓRICAS ===
encoder_municipio = TargetEncoder()
encoder_orgao = TargetEncoder()
encoder_grau = TargetEncoder()

df['Municipio_encoded'] = encoder_municipio.fit_transform(df['Municipio'], y)
df['Nome_orgao_encoded'] = encoder_orgao.fit_transform(df['Nome orgao'], y)
df['Grau_encoded'] = encoder_grau.fit_transform(df['Grau'], y)

X_cat = df[['Municipio_encoded', 'Nome_orgao_encoded', 'Grau_encoded']]

# === COMBINAR TODAS AS VARIÁVEIS ===
X = pd.concat([X_num, X_cat], axis=1)

# === DIVISÃO EM TREINO E TESTE ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === TREINAMENTO COM RANDOM FOREST ===
modelo_rf = RandomForestRegressor(random_state=42)
modelo_rf.fit(X_train, y_train)

# === AVALIAÇÃO DO MODELO ===
y_pred = modelo_rf.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)

print("\n=== DESEMPENHO DO MODELO APRIMORADO ===")
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R²: {r2:.2f}")

# === IMPORTÂNCIA DAS VARIÁVEIS ===
importancia_df = pd.DataFrame({
    'Variável': X.columns,
    'Importância': modelo_rf.feature_importances_
}).sort_values(by='Importância', ascending=False)

# === GRÁFICO DE IMPORTÂNCIA ===
plt.figure(figsize=(10, 6))
sns.barplot(x='Importância', y='Variável', data=importancia_df)
plt.title('Importância das Variáveis - Modelo Aprimorado')
plt.tight_layout()
plt.show()
