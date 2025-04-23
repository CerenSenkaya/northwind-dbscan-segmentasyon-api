from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from sqlalchemy import create_engine
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from kneed import KneeLocator

app = FastAPI()

# Veritabanı bağlantısı
user = "postgres"
password = "12345"
host = "localhost"
port = "5432"
database = "GYK-northwind"
engine = create_engine(f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}")

# DBSCAN fonksiyonu
def run_dbscan():
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

    # Ölçekleme
    X = df[["total_orders", "avg_order_value", "avg_items_per_order"]]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # eps belirleme
    neighbors = NearestNeighbors(n_neighbors=3).fit(X_scaled)
    distances, _ = neighbors.kneighbors(X_scaled)
    distances = sorted(distances[:, 2])
    kneedle = KneeLocator(range(len(distances)), distances, curve='convex', direction='increasing')
    optimal_eps = distances[kneedle.elbow]

    # DBSCAN
    dbscan = DBSCAN(eps=optimal_eps, min_samples=3)
    df["cluster"] = dbscan.fit_predict(X_scaled)

    return df

# Endpoint: /countries
@app.get("/countries")
def get_country_clusters():
    df = run_dbscan()
    return df.to_dict(orient="records")
