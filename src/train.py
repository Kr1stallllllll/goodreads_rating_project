import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import os

# 1. Загрузка данных
df = pd.read_csv('data/books.csv', on_bad_lines='skip')
df.columns = df.columns.str.strip()

# 2. Подготовка признаков (X) и цели (y)
features = ['num_pages', 'ratings_count', 'text_reviews_count']
target = 'average_rating'
X = df[features].fillna(0)
y = df[target]

# 3. Масштабирование
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Разделение данных
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Простое правило (Baseline)
y_mean = np.mean(y_train)
baseline_preds = [y_mean] * len(y_test)
mae_baseline = mean_absolute_error(y_test, baseline_preds)
print(f"MAE Базового правила: {mae_baseline:.4f}")

# Алгоритм (RandomForest) с кросс-валидацией
rf = RandomForestRegressor(n_estimators=100, random_state=42)
cv_scores = cross_val_score(rf, X_train, y_train, cv=5, scoring='neg_mean_absolute_error')
print(f"Средняя MAE на кросс-валидации (RandomForest): {-cv_scores.mean():.4f}")

# Обучаем RandomForest
rf.fit(X_train, y_train)
rf_preds = rf.predict(X_test)
mae_rf = mean_absolute_error(y_test, rf_preds)

# Визуализация ошибок
plt.figure(figsize=(10, 6))
plt.scatter(y_test, rf_preds, alpha=0.3)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
plt.xlabel("Реальный рейтинг")
plt.ylabel("Предсказанный рейтинг")
plt.title("Визуализация ошибок RandomForest")
plt.savefig('graphs/error_analysis.png')
print("График ошибок сохранен как graphs/error_analysis.png")

if mae_rf < mae_baseline:
    best_model = rf
    best_name = "RandomForestRegressor"
    best_mae = mae_rf
    final_preds = rf_preds
else:
    best_model = None
    best_name = "Базовое правило"
    best_mae = mae_baseline
    final_preds = baseline_preds

model_dir = 'models'
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# Сохранение
if best_model is not None:
    with open('models/best_model.pkl', 'wb') as f:
        pickle.dump(best_model, f)
    with open('models/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

# Итоговый отчёт
diff = abs(y_test - final_preds)
idx_max = diff.idxmax()
print("\n" + "="*40)
print(f"Лучшая модель — {best_name}.")
print(f"Её ключевая метрика на новых данных — {best_mae:.4f}.")
print(f"Чаще всего она путает {y_test.loc[idx_max]:.1f} и {final_preds[y_test.index.get_loc(idx_max)]:.1f}.")
print("="*40)