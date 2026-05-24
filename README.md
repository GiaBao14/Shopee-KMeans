# Shopee Product K-Means Analysis

Project này phân tích dữ liệu sản phẩm Shopee bằng thuật toán K-Means và hiển thị kết quả bằng dashboard Streamlit.

## Chức năng chính

- Xem bảng dữ liệu sản phẩm.
- Tìm kiếm sản phẩm theo tên, shop hoặc danh mục.
- Phân cụm sản phẩm theo giá bán.
- Phân cụm sản phẩm theo lượt mua và đánh giá.
- Phân cụm full feature với các đặc trưng:
  - `price`
  - `buyer_count`
  - `rating`
  - `category_encoded`
- Hiển thị kết quả phân cụm trên dashboard.

## Cấu trúc thư mục

```text
Project_Shoppe/
├── clustering/
│   ├── kmeans_price_cluster.py
│   ├── kmeans_product_segment.py
│   └── kmeans_full_feature.py
├── dashboard/
│   └── app.py
├── data/
│   └── data.csv
├── notebooks/
│   └── kmeans_analysis.ipynb
├── output/
│   ├── kmeans_price_result.csv
│   ├── kmeans_product_segment_result.csv
│   ├── kmeans_full_feature_result.csv
│   └── *.png
├── report.md
├── requirements.txt
└── README.md
```

## Cài đặt thư viện

```powershell
pip install -r requirements.txt
```

Nếu dùng môi trường ảo:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

## Chạy các file phân cụm

Phân cụm theo giá bán:

```powershell
python clustering/kmeans_price_cluster.py
```

Phân cụm sản phẩm hot, ít người mua, đánh giá cao:

```powershell
python clustering/kmeans_product_segment.py
```

Phân cụm full feature:

```powershell
python clustering/kmeans_full_feature.py
```

## Chạy dashboard

```powershell
streamlit run dashboard/app.py
```

Hoặc:

```powershell
python -m streamlit run dashboard/app.py
```

## Dữ liệu

File dữ liệu chính nằm tại:

```text
data/data.csv
```

Các cột chính:

- `product_name`: tên sản phẩm
- `shop_name`: tên shop
- `price`: giá bán
- `category`: danh mục sản phẩm
- `buyer_count`: lượt mua
- `rating`: điểm đánh giá

## Kết quả đầu ra

Các file kết quả được lưu trong thư mục `output/`, gồm:

- `kmeans_price_result.csv`
- `kmeans_product_segment_result.csv`
- `kmeans_full_feature_result.csv`
- các ảnh biểu đồ `.png`

## Ghi chú

Nếu chạy dashboard nhưng chưa có đủ file trong `output/`, hãy chạy các file trong thư mục `clustering/` trước.
