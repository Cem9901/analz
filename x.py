from lightgbm import LGBMRegressor
from sklearn.model_selection import RandomizedSearchCV, train_test_split
import numpy as np
import pandas as pd

def random_search_and_train(df, target):
    # İlk olarak random_search fonksiyonunu kullanarak en iyi parametreleri ve modeli bulalım
    print("Random Search ile En İyi Parametreleri Bulma ve Model Eğitme:")
    best_model, feature_importances = random_search(df.drop(target, axis=1), df[target])

    # Ardından, train_and_predict fonksiyonunu kullanarak bu en iyi modeli eğitip değerlendirelim
    print("\nEn İyi Modelin Performansını Değerlendirme:")
    result_df, trained_model = train_and_predict(best_model, df, target)

    # En iyi modelin özellik önem skorlarını gösterme
    print("\nEn İyi Modelin Özellik Önem Skorları:")
    print(feature_importances)

    return result_df, trained_model

# Örnek kullanım:
# df ve target değerlerini uygun şekilde belirtmelisiniz.
result_df, final_model = random_search_and_train(df, 'target_column')
