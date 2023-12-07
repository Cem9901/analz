# Gerekli kütüphaneleri içe aktarın
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd

# Veri setini yükleyin (örnek olarak kullanılan bir CSV dosyası)
data = pd.read_csv('veri.csv')

# Bağımlı değişkeni ve özellikleri seçin
X = data.drop('hedef_sutun', axis=1)
y = data['hedef_sutun']

# Veriyi eğitim ve test setlerine bölelim
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# XGBoost Regresyon modelini oluşturun
model = xgb.XGBRegressor(objective='reg:squarederror', seed=42)

# Modeli eğitin
model.fit(X_train, y_train)

# Eğitim seti üzerinde tahmin yapın
y_train_pred = model.predict(X_train)

# Test seti üzerinde tahmin yapın
y_test_pred = model.predict(X_test)

# Eğitim seti üzerinde aşırı uyma kontrolü
train_mse = mean_squared_error(y_train, y_train_pred)
print(f'Eğitim seti MSE: {train_mse}')

# Test seti üzerinde aşırı uyma kontrolü
test_mse = mean_squared_error(y_test, y_test_pred)
print(f'Test seti MSE: {test_mse}')

# K-fold çapraz doğrulama kullanarak model performansını değerlendirin
cv_scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
cv_rmse_scores = np.sqrt(-cv_scores)

# K-fold çapraz doğrulama sonuçlarını yazdır
print("K-fold Çapraz Doğrulama RMSE Skorları:")
for i, score in enumerate(cv_rmse_scores):
    print(f'Fold-{i+1}: {score}')

# Hiperparametre ayarlaması yaparak aşırı uyma kontrolü
# Örnek olarak, ağaç sayısını ve derinliği kontrol edelim
model_tuned = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=200, max_depth=3, seed=42)
model_tuned.fit(X_train, y_train)

# Modeli değerlendirin
y_test_pred_tuned = model_tuned.predict(X_test)
test_mse_tuned = mean_squared_error(y_test, y_test_pred_tuned)
print(f'Tuned Model Test MSE: {test_mse_tuned}')














# Hiperparametre Ayarlamaları ile Aşırı Uyuma Kontrolü
# Örneğin, ağaç sayısı ve ağaç derinliğini kontrol edelim
model_tuned = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, max_depth=3, random_state=42)
model_tuned.fit(X_train, y_train)

# Modelin performansını değerlendirin
y_test_pred_tuned = model_tuned.predict(X_test)
test_mse_tuned = mean_squared_error(y_test, y_test_pred_tuned)
test_rmse_tuned = np.sqrt(test_mse_tuned)
test_mae_tuned = mean_absolute_error(y_test, y_test_pred_tuned)
test_r2_tuned = r2_score(y_test, y_test_pred_tuned)

print("Tuned Model Performans Metrikleri:")
print(f'Test MSE: {test_mse_tuned}')
print(f'Test RMSE: {test_rmse_tuned}')
print(f'Test MAE: {test_mae_tuned}')
print(f'Test R-squared: {test_r2_tuned}')




# K-fold Çapraz Doğrulama ile Performansı Değerlendirme
# K-fold çapraz doğrulama için bir KFold nesnesi oluşturun
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# XGBoost modelinizi kullanarak RMSE skorlarını alın
cv_scores = cross_val_score(model, X, y, cv=kf, scoring='neg_mean_squared_error')

# Skorları pozitif hale getirin ve karekök alarak RMSE'ye dönüştürün
cv_rmse_scores = np.sqrt(-cv_scores)

# K-fold çapraz doğrulama sonuçlarını yazdır
print("K-fold Çapraz Doğrulama RMSE Skorları (Orijinal Model):")
for i, score in enumerate(cv_rmse_scores):
    print(f'Fold-{i+1}: {score}')

# Tuned Model ile K-fold Çapraz Doğrulama
cv_scores_tuned = cross_val_score(model_tuned, X, y, cv=kf, scoring='neg_mean_squared_error')
cv_rmse_scores_tuned = np.sqrt(-cv_scores_tuned)

print("\nK-fold Çapraz Doğrulama RMSE Skorları (Tuned Model):")
for i, score in enumerate(cv_rmse_scores_tuned):
    print(f'Fold-{i+1}: {score}')

