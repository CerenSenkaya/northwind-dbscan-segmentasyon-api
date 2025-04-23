# Ülke bazlı satış deseni analizi: Sipariş alışkanlıklarına göre ülkeleri grupla ve sıra dışı olanları bulma

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from kneed import KneeLocator
import warnings
warnings.filterwarnings("ignore")

# Veritabanı bağlantısı
user = "postgres"
password = "12345"
host = "localhost"
port = "5432"
database = "GYK-northwind"

engine = create_engine(f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}")

# SQL sorgusu
query = """
select 
    c.country,
    count(distinct o.order_id) as total_orders,
    avg(od.unit_price * od.quantity) as avg_order_value,
    sum(od.quantity)::float / count(distinct o.order_id) as avg_items_per_order
from customers c
join orders o on c.customer_id = o.customer_id
join order_details od on o.order_id = od.order_id
group by c.country
"""

df = pd.read_sql_query(query, engine)
print("Ülke bazlı sipariş verileri:\n", df.head())

# Özellik vektörü
X = df[["total_orders", "avg_order_value", "avg_items_per_order"]]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# eps + min_samples 
def optimize_dbscan(X_scaled, min_samples_list=[2, 3, 4, 5]):
    for min_samples in min_samples_list:
        neighbors = NearestNeighbors(n_neighbors=min_samples).fit(X_scaled)
        distances, _ = neighbors.kneighbors(X_scaled)
        distances = np.sort(distances[:, min_samples - 1])
        kneedle = KneeLocator(range(len(distances)), distances, curve="convex", direction="increasing")
        optimal_eps = distances[kneedle.elbow]
        
        print(f"[Min_samples={min_samples}] Optimal eps: {optimal_eps:.2f}")

        plt.figure(figsize=(8, 4))
        plt.plot(distances)
        plt.axvline(x=kneedle.elbow, color='r', linestyle='--', label=f'eps = {optimal_eps:.2f}')
        plt.title(f"Elbow plot (min_samples = {min_samples})")
        plt.xlabel("Sıralı noktalar")
        plt.ylabel("Mesafe")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

        yield min_samples, optimal_eps

# eps & min_samples seçeneklerini deneme
results = list(optimize_dbscan(X_scaled, min_samples_list=[2, 3, 4]))

# En iyi sonucu manuel seç (örnek: min_samples=3)
best_min_samples, best_eps = results[1]  # burada [1] → min_samples=3 için

# DBSCAN uygulaması
dbscan = DBSCAN(eps=best_eps, min_samples=best_min_samples)
df["cluster"] = dbscan.fit_predict(X_scaled)

# Görselleştirme
plt.figure(figsize=(10, 6))
plt.scatter(df["total_orders"], df["avg_order_value"], c=df["cluster"], cmap="plasma", s=100)
plt.xlabel("Toplam Sipariş Sayısı")
plt.ylabel("Ortalama Sipariş Tutarı")
plt.title("Ülkelere Göre Satış Deseni Segmentasyonu (DBSCAN)")
plt.colorbar(label="Küme No")
plt.grid(True)
plt.tight_layout()
plt.show()

# Aykırı ülkeleri yazdırma
outliers = df[df["cluster"] == -1]
print("Aykırı ülkeler:")
print(outliers[["country", "total_orders", "avg_order_value", "avg_items_per_order"]])
