import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from tabulate import tabulate

# ===========================================================
# 1. ВИКОНАННЯ ОБРАХУНКІВ
# ===========================================================

# Генеруємо тестові дані
countries = [f'Country_{i+1}' for i in range(50)]
years = list(range(2005, 2020))

data_list = []
np.random.seed(42)

for country in countries:
    for year in years:
        data_list.append({
            'Country': country,
            'Year': year,
            'GrainProd': np.random.uniform(100, 1000),
            'FoodImport': np.random.uniform(0, 50),
            'Calories': np.random.uniform(2000, 3500),
            'Undernourishment': np.random.uniform(1, 30),
            'FoodCost': np.random.uniform(5, 30),
            'GFSI': np.random.uniform(40, 90)
        })

data = pd.DataFrame(data_list)

# Додаємо пропуски
for col in ['GrainProd','FoodImport','Calories','Undernourishment','FoodCost','GFSI']:
    data.loc[data.sample(frac=0.05).index, col] = np.nan

# Заповнення пропусків
data = data.sort_values(['Country', 'Year'])
cols = ['GrainProd','FoodImport','Calories','Undernourishment','FoodCost','GFSI']
data[cols] = data.groupby('Country')[cols].transform(
    lambda x: x.interpolate().fillna(x.mean())
)

print("\n=== [TABLE] Дані після інтерполяції (перші 10) ===")
print(tabulate(data.head(10), headers='keys', tablefmt='grid'))

# Нормалізація
scaler = MinMaxScaler()
data[cols] = scaler.fit_transform(data[cols])

# Побудова індексу
data['Undernourishment'] = 1 - data['Undernourishment']
data['FoodCost'] = 1 - data['FoodCost']
data['FoodSecurityIndex'] = data[cols].mean(axis=1)

# Вивід 10 перших
print("\n=== [TABLE] Індекс продовольчої безпеки (перші 10) ===")
print(tabulate(data[['Country','Year','FoodSecurityIndex']].head(10), headers='keys', tablefmt='grid'))


# ===========================================================
# 2. ПЕРЕВІРКА НА ЧУТЛИВІСТЬ ТА АДЕКВАТНІСТЬ
# ===========================================================

# Кореляції між індикаторами
corr_matrix = data[cols].corr()

print("\n=== [CHECK] Кореляційна матриця показників ===")
print(tabulate(corr_matrix, headers='keys', tablefmt='grid'))

# Аналіз чутливості – зміна ваг
weights_sets = {
    "рівні ваги": np.array([1,1,1,1,1,1]),
    "виробництво важливіше": np.array([2,1,1,1,1,1]),
    "витрати важливіші": np.array([1,1,1,1,2,1])
}

print("\n=== [CHECK] Аналіз чутливості моделі ===")
for name, w in weights_sets.items():
    w = w / w.sum()  # нормалізація ваг
    data[f'FSI_{name}'] = (data[cols] * w).sum(axis=1)
    print(f"- Обчислено варіант індексу: {name}")

print("\n=== [TABLE] Порівняння трьох індексів (перші 10) ===")
print(tabulate(
    data[['Country','Year','FoodSecurityIndex','FSI_рівні ваги',
          'FSI_виробництво важливіше','FSI_витрати важливіші']].head(10),
    headers='keys', tablefmt='grid'
))


# ===========================================================
# 3. ІМІТАЦІЯ ПУБЛІКАЦІЇ В ZENODO
# ===========================================================

print("\n=== [UPLOAD] Імітація публікації в Zenodo ===")
print("Файли мали б бути викладені на Zenodo через API.")
print("У звіті вставити скрін сторінки завантаження на Zenodo.")


# ===========================================================
# 4. АНАЛІЗ ТА ГРАФІЧНИЙ ДАШБОРД
# ===========================================================

# PCA
x = data[cols]
pca = PCA(n_components=2)
components = pca.fit_transform(x)
data['PC1'] = components[:,0]
data['PC2'] = components[:,1]

print("\n=== [TABLE] PCA результат (перші 10) ===")
print(tabulate(data[['Country','Year','PC1','PC2']].head(10), headers='keys', tablefmt='grid'))

# --- Графіки ---

# 1. Рейтинг країн 2019
fsi_2019 = data[data['Year'] == 2019].sort_values('FoodSecurityIndex', ascending=False)

plt.figure(figsize=(14,6))
sns.barplot(x='Country', y='FoodSecurityIndex', data=fsi_2019)
plt.xticks(rotation=90)
plt.title('Food Security Index by Country (2019)')
plt.tight_layout()
plt.savefig("01_FSI_2019.png")
plt.close()

# 2. Кореляційна матриця
plt.figure(figsize=(8,6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title("Correlation Matrix of Indicators")
plt.tight_layout()
plt.savefig("02_CorrelationMatrix.png")
plt.close()

# 3. PCA Scatter
plt.figure(figsize=(10,6))
sns.scatterplot(x='PC1', y='PC2', hue='FoodSecurityIndex', data=data, palette="viridis")
plt.title("PCA of Indicators")
plt.tight_layout()
plt.savefig("03_PCA_Scatter.png")
plt.close()

print("\n=== ГОТОВО ===")
print("Таблиці виведені у консоль, графіки збережені у файли:")
print("- 01_FSI_2019.png")
print("- 02_CorrelationMatrix.png")
print("- 03_PCA_Scatter.png")
