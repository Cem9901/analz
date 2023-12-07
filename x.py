# Özellik önemini alın
feature_importance = xg_reg.feature_importances_

# Özellik isimlerini ve önem sıralarını bir veri çerçevesine yerleştirin
feature_importance_df = pd.DataFrame({'Feature': X_train.columns, 'Importance': feature_importance})

# Özellikleri önem sırasına göre sıralayın
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Özellik önemini görselleştirin
plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df, palette='viridis')
plt.title('XGBoost - Özellik Önemi')
plt.xlabel('Özellik Önemi')
plt.ylabel('Özellik')
plt.show()

# Kümülatif önem sıralamasını görselleştirin
cumulative_importance = feature_importance_df['Importance'].cumsum()

plt.figure(figsize=(10, 6))
plt.plot(range(1, len(cumulative_importance) + 1), cumulative_importance, marker='o', linestyle='--', color='b')
plt.title('XGBoost - Kümülatif Özellik Önemi')
plt.xlabel('Toplam Özellik Sayısı')
plt.ylabel('Kümülatif Özellik Önemi')
plt.grid(True)
plt.show()
