# Northwind DBSCAN Segmentasyon Projesi

Bu proje, Northwind veritabanı üzerinde DBSCAN algoritması kullanılarak:

- Müşteri segmentasyonu  
- Ürün segmentasyonu  
- Tedarikçi segmentasyonu  
- Ülke bazlı satış alışkanlıklarının analizi  

yapmak üzere geliştirilmiştir. Ayrıca FastAPI ile bu analizler birer **API servisi** haline getirilmiştir.

---

##  Kullanılan Teknolojiler

- Python 3
- PostgreSQL (Northwind verisi)
- Pandas, Scikit-learn, Matplotlib
- FastAPI
- DBSCAN (Yoğunluk Tabanlı Kümeleme)
- Uvicorn (API sunucusu)

---

## API Endpointleri

| Endpoint            | Açıklama                                 |
|---------------------|------------------------------------------|
| `/products`         | Ürünleri satış davranışına göre gruplar |
| `/suppliers`        | Tedarikçileri ürün performansına göre gruplar |
| `/countries`        | Ülkeleri sipariş alışkanlıklarına göre gruplar |

Swagger arayüzü: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## 🚀 Projeyi Çalıştırma

```bash
uvicorn product_supplier_api:app --reload


