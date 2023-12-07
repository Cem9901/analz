# Gerekli kütüphaneleri yükleyin
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Veriyi yükleyin veya oluşturun
# Örnek olarak rastgele bir veri seti oluşturuyoruz
np.random.seed(42)
X = np.random.rand(100, 1) * 10
y = 2 * X.squeeze() + 1 + np.random.randn(100) * 2

# Veriyi eğitim ve test setlerine ayırın
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# XGBoost regresyon modelini oluşturun ve eğitin
model = xgb.XGBRegressor()
model.fit(X_train, y_train)

# Modelin tahminleri
y_pred = model.predict(X_test)

# 1. Modelin Performansını Değerlendirme
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'MSE: {mse:.2f}')
print(f'RMSE: {rmse:.2f}')
print(f'MAE: {mae:.2f}')
print(f'R-squared: {r2:.2f}')

# 2. Model Ayarlama (Grid Search kullanarak)
param_grid = {
    'learning_rate': [0.01, 0.1, 0.2],
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7]
}

grid_search = GridSearchCV(xgb.XGBRegressor(), param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)

# En iyi parametreleri bulma
best_params = grid_search.best_params_

# Yeni modeli en iyi parametrelerle oluşturma
best_model = xgb.XGBRegressor(**best_params)
best_model.fit(X_train, y_train)

# Yeni modelin performansını değerlendirme
y_pred_best = best_model.predict(X_test)

mse_best = mean_squared_error(y_test, y_pred_best)
rmse_best = np.sqrt(mse_best)
mae_best = mean_absolute_error(y_test, y_pred_best)
r2_best = r2_score(y_test, y_pred_best)

print('\nSonrasında Grid Search ile Ayarlanmış Model:')
print(f'MSE: {mse_best:.2f}')
print(f'RMSE: {rmse_best:.2f}')
print(f'MAE: {mae_best:.2f}')
print(f'R-squared: {r2_best:.2f}')

# 3. Modelin Şeffaflığını ve Yorumlanabilirliğini İnceleme
# Özelliklerin önem sıralamalarını görselleştirme
xgb.plot_importance(model)
plt.show()

# 4. Aykırı Değerleri İnceleme ve Eleme (Opsiyonel)
# Aykırı değerleri belirleme
outliers = np.abs(y_train - model.predict(X_train)) > 3 * rmse

# Aykırı değerleri eleme
X_train_no_outliers = X_train[~outliers]
y_train_no_outliers = y_train[~outliers]

# Yeni modeli eğitme
model_no_outliers = xgb.XGBRegressor()
model_no_outliers.fit(X_train_no_outliers, y_train_no_outliers)

# 5. Cross-Validation Kullanma (Opsiyonel)
from sklearn.model_selection import cross_val_score

cv_scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
cv_rmse_scores = np.sqrt(-cv_scores)
print('\nCross-Validation RMSE Scores:', cv_rmse_scores)

# 6. Farklı Modelleri Karşılaştırma (Opsiyonel)
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

# Farklı modelleri oluşturma
rf_model = RandomForestRegressor()
lr_model = LinearRegression()

# Modelleri eğitme
rf_model.fit(X_train, y_train)
lr_model.fit(X_train, y_train)

# Modellerin performansını değerlendirme
y_pred_rf = rf_model.predict(X_test)
y_pred_lr = lr_model.predict(X_test)

mse_rf = mean_squared_error(y_test, y_pred_rf)
mse_lr = mean_squared_error(y_test, y_pred_lr)

print('\nRandom Forest Model MSE:', mse_rf)
print('Linear Regression Model MSE:', mse_lr)

# 7. Sonuçları İlgili Taraflarla Paylaşma
# Modelinizi raporlayarak ve sonuçları paylaşarak ilgili tarafları bilgilendirin

# 8. Geriye Dönük Analiz ve Öğrenme
# Modelin yanlış tahminlerini inceleyerek ve öğrenerek gelecekteki model geliştirmeleri için bilgi edinme

# 9. Modeli Dağıtma (Deployment)
# Modelinizi gerçek dünya verileri üzerinde test edin ve gerektiğinde ayarlamalar yaparak kullanıma alın
