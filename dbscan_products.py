# Ürün segmentasyonu: Benzer sipariş geçmişine göre ürünleri gruplayıp ve aykırı olanları bulma

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from kneed import KneeLocator

# Veritabanı bağlantı bilgileri
user = "postgres"
password = "12345"
host = "localhost"
port = "5432"
database = "GYK-northwind"  # Veya "GYK-northwind" ise o şekilde bırak

# Bağlantıyı oluştur
engine = create_engine(f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}")

# SQL sorgusu: Ürün bazlı satış metrikleri
query = """
select 
    p.product_id,
    p.product_name,
    avg(od.unit_price) as avg_unit_price,
    count(*) as sales_count,
    avg(od.quantity) as avg_quantity_per_order,
    count(distinct o.customer_id) as distinct_customers
from products p
join order_details od on p.product_id = od.product_id
join orders o on od.order_id = o.order_id
group by p.product_id, p.product_name
"""

# Veriyi oku
df = pd.read_sql_query(query, engine)
print("İlk 5 ürün:\n", df.head())

# Özellik vektörleri
X = df[["avg_unit_price", "sales_count", "avg_quantity_per_order", "distinct_customers"]]

# Veriyi ölçeklendir
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# En uygun eps değeri bulmak için fonksiyon
def find_optimal_eps(X_scaled, min_samples=3):
    neighbors = NearestNeighbors(n_neighbors=min_samples).fit(X_scaled)
    distances, _ = neighbors.kneighbors(X_scaled)
    distances = np.sort(distances[:, min_samples - 1])
    kneedle = KneeLocator(range(len(distances)), distances, curve="convex", direction="increasing")
    optimal_eps = distances[kneedle.elbow]

    plt.figure(figsize=(10, 6))
    plt.plot(distances)
    plt.axvline(x=kneedle.elbow, color='r', linestyle='--', label=f'Optimal eps: {optimal_eps:.2f}')
    plt.title("Knee yöntemi ile optimal eps seçimi")
    plt.xlabel("Sıralı noktalar")
    plt.ylabel("Mesafe")
    plt.grid(True)
    plt.legend()
    plt.show()

    return optimal_eps

# eps bul, DBSCAN uygula
optimal_eps = find_optimal_eps(X_scaled)
dbscan = DBSCAN(eps=optimal_eps, min_samples=3)
df["cluster"] = dbscan.fit_predict(X_scaled)

# Kümeleme görselleştirme
plt.figure(figsize=(10, 6))
plt.scatter(df["sales_count"], df["distinct_customers"], c=df["cluster"], cmap="plasma", s=60)
plt.xlabel("Satış Sıklığı")
plt.ylabel("Farklı Müşteri Sayısı")
plt.title("Ürün Segmentasyonu (DBSCAN)")
plt.colorbar(label="Küme No")
plt.grid(True)
plt.show()

# Aykırı ürünler
outliers = df[df["cluster"] == -1]
print(f"Aykırı ürün sayısı: {len(outliers)}")
print(outliers[["product_id", "product_name", "sales_count", "distinct_customers"]])
