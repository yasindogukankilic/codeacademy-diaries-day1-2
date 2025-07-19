import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# 1. Veriyi CSV dosyasından oku
reviews = pd.read_csv("./data/reviews.csv")

# 2. Sütun adlarını ve genel bilgi özetini yazdır
print(reviews.columns)
print(reviews.info())

# 3. 'recommended' sütunundaki değerleri kontrol et
recommended = reviews["recommended"]

# 4. Boolean değerleri binary (0-1) formatına dönüştürmek için sözlük oluştur
binary_dict = {True: 1, False: 0}

# 5. 'recommended' sütununu binary formatına çevir
reviews["recommended"] = reviews["recommended"].map(binary_dict)

# 6. Dönüştürülmüş 'recommended' sütununu yazdır
print(reviews["recommended"])

# 7. 'rating' sütununu kontrol et
rating = reviews["rating"]
print(rating)

# 8. Rating değerlerini sıralı (ordinal) olarak sayısal değerlere eşle
rating_binary_dict = {"Loved it":5, "Liked it": 4, "Was okay": 3, "Not great": 2,"Hated it": 1}

# 9. 'rating' sütununu sayısal değerlere dönüştür
reviews["rating"] = reviews["rating"].map(rating_binary_dict)

# 10. Dönüştürülmüş 'rating' sütununu yazdır
print(reviews["rating"])

# 11. 'Department Name' sütunundaki farklı kategori sayısını hesapla
num_categories = reviews["department_name"].nunique()
print(num_categories)

# 12. 'Department Name' sütununu one-hot encoding ile binary sütunlara ayır
one_hot = pd.get_dummies(reviews['department_name'])

# 13. Yeni oluşturulan one-hot sütunlarını orijinal DataFrame'e ekle
reviews = reviews.join(one_hot)

# 14. DataFrame hakkında güncel bilgi yazdır (yeni sütunlarla birlikte)
print(reviews.info())

# 15. 'review_date' sütununu datetime formatına dönüştür
reviews['review_date'] = pd.to_datetime(reviews['review_date'])

# 16. 'review_date' sütununun veri tipini kontrol et
print(reviews['review_date'].dtype)

# 17. Modellemeye dahil edilecek sayısal sütunları seç
reviews = reviews[['clothing_id', 'age', 'recommended', 'rating', 
                   'Bottoms', 'Dresses', 'Intimate', 'Jackets', 'Tops', 'Trend']].copy()

# 18. 'clothing_id' sütununu indeks olarak ayarla
reviews = reviews.set_index('clothing_id')

# 19. StandardScaler nesnesi oluştur
scaler = StandardScaler()

# 20. Veriyi ölçeklendir (mean=0, std=1 olacak şekilde)
scaled_data = scaler.fit_transform(reviews)

# 21. Ölçeklenmiş veriyi yazdır (NumPy array olarak)
print(scaled_data)
