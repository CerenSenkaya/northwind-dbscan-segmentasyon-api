# FastAPI: Products & Suppliers clustering using DBSCAN
# /products and /suppliers endpoints

from fastapi import FastAPI
from sqlalchemy import create_engine
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from kneed import KneeLocator
app = FastAPI()

app = FastAPI()

# Veritabanı bağlantısı
user = "postgres"
password = "12345"
host = "localhost"
port = "5432"
database = "GYK-northwind"

engine = create_engine(f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}")

# eps + DBSCAN fonksiyonu
def run_dbscan(df, feature_cols, min_samples=3):
    X = df[feature_cols]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    neighbors = NearestNeighbors(n_neighbors=min_samples).fit(X_scaled)
    distances, _ = neighbors.kneighbors(X_scaled)
    distances = sorted(distances[:, min_samples - 1])
    kneedle = KneeLocator(range(len(distances)), distances, curve='convex', direction='increasing')
    eps = distances[kneedle.elbow]

    model = DBSCAN(eps=eps, min_samples=min_samples)
    df["cluster"] = model.fit_predict(X_scaled)
    return df

# /products endpoint
@app.get("/products")
def product_clusters():
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
    df = pd.read_sql_query(query, engine)
    df = run_dbscan(df, ["avg_unit_price", "sales_count", "avg_quantity_per_order", "distinct_customers"])
    return df.to_dict(orient="records")

# /suppliers endpoint
@app.get("/suppliers")
def supplier_clusters():
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
    df = run_dbscan(df, ["product_count", "total_quantity_sold", "avg_unit_price", "avg_customer_count"])
    return df.to_dict(orient="records")
