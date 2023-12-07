# Hata dağılım grafiği
plt.figure(figsize=(10, 6))
plt.scatter(y_test, errors, alpha=0.5)
plt.title('Hata Dağılım Grafiği')
plt.xlabel('Gerçek Değerler')
plt.ylabel('Hatalar')
plt.show()



# Gerçek değerler ile tahmin edilen değerlerin karşılaştırılması
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.title('Gerçek Değerler ve Tahminlerin Karşılaştırılması')
plt.xlabel('Gerçek Değerler')
plt.ylabel('Tahmin Edilen Değerler')
plt.show()
