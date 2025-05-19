# modelagem_preditiva.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Carregar os dados tratados
df = pd.read_csv("../dados/tjsp_processos_tratado.csv", sep=";")

# Selecionar variável alvo e preditoras (numéricas)
y = df['TPSent_12_meses_num']
X = df[[
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

# Dividir dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modelos a serem testados
modelos = {
    'Regressão Linear': LinearRegression(),
    'Arvore de Decisão': DecisionTreeRegressor(random_state=42),
    'Random Forest': RandomForestRegressor(random_state=42)
}

# Avaliação dos modelos
resultados = []

for nome, modelo in modelos.items():
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, y_pred)

    resultados.append({
        'Modelo': nome,
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2
    })

# Mostrar resultados
resultados_df = pd.DataFrame(resultados)
print("\nResultados da Modelagem:")
print(resultados_df.sort_values(by='R2', ascending=False))

# Gráfico comparativo de desempenho
plt.figure(figsize=(10, 5))
sns.barplot(data=resultados_df, x='Modelo', y='R2')
plt.title('Comparativo de R² entre Modelos')
plt.ylabel('Coeficiente de Determinação (R²)')
plt.xlabel('Modelo')
plt.ylim(0, 1)
plt.tight_layout()
plt.show()

# Ajuste fino de hiperparâmetros da Random Forest com GridSearchCV
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(estimator=RandomForestRegressor(random_state=42),
                           param_grid=param_grid,
                           cv=5,
                           scoring='neg_mean_absolute_error',
                           n_jobs=-1,
                           verbose=1)

grid_search.fit(X_train, y_train)

# Melhor modelo
best_model = grid_search.best_estimator_
print("\nMelhores hiperparâmetros encontrados:")
print(grid_search.best_params_)

# Avaliação do modelo ajustado
y_pred_best = best_model.predict(X_test)
mae_best = mean_absolute_error(y_test, y_pred_best)
rmse_best = mean_squared_error(y_test, y_pred_best) ** 0.5
r2_best = r2_score(y_test, y_pred_best)

print("\nResultados do modelo ajustado:")
print(f"MAE: {mae_best:.2f}")
print(f"RMSE: {rmse_best:.2f}")
print(f"R²: {r2_best:.2f}")

# Gráfico de importância das variáveis
importances = best_model.feature_importances_
features = X.columns
importancia_df = pd.DataFrame({'Variável': features, 'Importância': importances}).sort_values(by='Importância', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importância', y='Variável', data=importancia_df, palette='viridis')
plt.title('Importância das Variáveis - Random Forest Ajustado')
plt.tight_layout()
plt.savefig("../graficos/importancia_variaveis_rf_ajustado.png", dpi=300)
plt.show()

# Gráfico comparando modelos antes e depois do ajuste (R²)
resultados_df = pd.DataFrame({
    'Modelo': ['Random Forest (Padrão)', 'Random Forest (Ajustado)'],
    'R²': [0.416838, r2_best]
})

plt.figure(figsize=(8, 5))
sns.barplot(x='Modelo', y='R²', data=resultados_df, palette='Blues_d')
plt.title('Comparação de Desempenho do Random Forest')
plt.ylim(0, 1)
plt.ylabel('Coeficiente de Determinação (R²)')
plt.tight_layout()
plt.savefig("../graficos/comparacao_modelos_rf.png", dpi=300)
plt.show()