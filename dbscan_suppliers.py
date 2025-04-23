# Tedarikçi segmentasyonu: Ürün performansına göre tedarikçileri gruplayıp ve aykırı olanları bulma

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from kneed import KneeLocator

# Veritabanı bağlantısı
user = "postgres"
password = "12345"
host = "localhost"
port = "5432"
database = "GYK-northwind"

engine = create_engine(f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}")

# SQL sorgusu: Tedarikçi bazlı özellikler
query = """
select 
    s.supplier_id,
    s.company_name,
    count(distinct p.product_id) as product_count,
    sum(od.quantity) as total_quantity_sold,
    avg(od.unit_price) as avg_unit_price,
    avg(sub.customer_count) as avg_customer_count
from suppliers s
join products p on s.supplier_id = p.supplier_id
join order_details od on p.product_id = od.product_id
join orders o on od.order_id = o.order_id
join (
    select 
        p2.product_id,
        count(distinct o2.customer_id) as customer_count
    from products p2
    join order_details od2 on p2.product_id = od2.product_id
    join orders o2 on od2.order_id = o2.order_id
    group by p2.product_id
) sub on sub.product_id = p.product_id
group by s.supplier_id, s.company_name
"""

df = pd.read_sql_query(query, engine)
print("Tedarikçi verileri:\n", df.head())

# Özellik seçimi ve ölçekleme
X = df[["product_count", "total_quantity_sold", "avg_unit_price", "avg_customer_count"]]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Optimal eps bulma fonksiyonu
def find_optimal_eps(X_scaled, min_samples=2):
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

# eps bul ve DBSCAN uygula
optimal_eps = find_optimal_eps(X_scaled)
dbscan = DBSCAN(eps=optimal_eps, min_samples=2)
df["cluster"] = dbscan.fit_predict(X_scaled)

# Kümeleme görselleştirme
plt.figure(figsize=(10, 6))
plt.scatter(df["product_count"], df["total_quantity_sold"], c=df["cluster"], cmap="plasma", s=60)
plt.xlabel("Ürün Sayısı")
plt.ylabel("Toplam Satış Miktarı")
plt.title("Tedarikçi Segmentasyonu (DBSCAN)")
plt.colorbar(label="Küme No")
plt.grid(True)
plt.show()

# Aykırı tedarikçileri yazdır
outliers = df[df["cluster"] == -1]
print(f"Aykırı tedarikçi sayısı: {len(outliers)}")
print(outliers[["supplier_id", "company_name", "product_count", "total_quantity_sold"]])
