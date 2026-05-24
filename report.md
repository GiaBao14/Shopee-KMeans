# Báo Cáo Phân Tích Sản Phẩm Shopee Bằng K-Means

## 1. Mục tiêu

Project sử dụng thuật toán K-Means để phân cụm sản phẩm Shopee dựa trên các thông tin như giá bán, lượt mua, đánh giá và danh mục sản phẩm. Kết quả được hiển thị bằng dashboard để dễ quan sát và so sánh.

## 2. Dữ liệu

Dữ liệu được lưu trong file `data/data.csv`.

Các cột chính:

| Cột | Ý nghĩa |
| --- | --- |
| `product_name` | Tên sản phẩm |
| `shop_name` | Tên shop |
| `price` | Giá bán |
| `category` | Danh mục sản phẩm |
| `buyer_count` | Số lượt mua |
| `rating` | Điểm đánh giá |

## 3. Tiền xử lý dữ liệu

Các bước xử lý chính:

- Đọc dữ liệu từ file CSV.
- Xóa dữ liệu trùng lặp.
- Chuyển các cột số như `price`, `buyer_count`, `rating` về kiểu numeric.
- Kiểm tra dữ liệu không hợp lệ.
- Mã hóa cột `category` thành `category_encoded` khi chạy K-Means full feature.
- Chuẩn hóa dữ liệu bằng `MinMaxScaler` trước khi đưa vào K-Means.

## 4. Các mô hình K-Means

### 4.1. K-Means theo giá bán

File thực hiện:

```text
clustering/kmeans_price_cluster.py
```

Feature sử dụng:

```text
price
```

Mục đích:

- Chia sản phẩm thành các nhóm giá.
- Xác định nhóm sản phẩm giá thấp, trung bình và cao.
- Hiển thị top 6 sản phẩm có giá bán cao nhất.

Kết quả lưu tại:

```text
output/kmeans_price_result.csv
```

### 4.2. K-Means theo lượt mua và đánh giá

File thực hiện:

```text
clustering/kmeans_product_segment.py
```

Feature sử dụng:

```text
buyer_count
rating
```

Mục đích:

- Nhận diện sản phẩm hot.
- Nhận diện sản phẩm ít người mua.
- Nhận diện sản phẩm có đánh giá cao.

Kết quả lưu tại:

```text
output/kmeans_product_segment_result.csv
```

### 4.3. K-Means full feature

File thực hiện:

```text
clustering/kmeans_full_feature.py
```

Feature sử dụng:

```text
price
buyer_count
rating
category_encoded
```

Mục đích:

- Phân cụm sản phẩm dựa trên nhiều đặc trưng cùng lúc.
- Kết hợp yếu tố giá, lượt mua, đánh giá và danh mục.
- Giúp quan sát nhóm sản phẩm theo hành vi tổng quát hơn.

Kết quả lưu tại:

```text
output/kmeans_full_feature_result.csv
```

## 5. Dashboard

Dashboard được xây dựng bằng Streamlit tại:

```text
dashboard/app.py
```

Các phần chính:

- Bảng dữ liệu sản phẩm.
- Tìm kiếm sản phẩm.
- Biểu đồ K-Means theo giá bán.
- Biểu đồ thống kê sản phẩm hot, ít người mua, đánh giá cao.
- Kết quả K-Means full feature.

Lệnh chạy:

```powershell
streamlit run dashboard/app.py
```

## 6. Nhận xét

K-Means giúp chia sản phẩm thành các nhóm có đặc điểm tương đồng. Với dữ liệu Shopee, các nhóm có thể hỗ trợ việc phân tích:

- Sản phẩm có giá cao.
- Sản phẩm bán chạy.
- Sản phẩm ít người mua.
- Sản phẩm có đánh giá tốt.
- Các nhóm sản phẩm có hành vi tương tự nhau.

## 7. Hạn chế

- Dữ liệu còn nhỏ nên kết quả phân cụm chỉ mang tính minh họa.
- K-Means nhạy với cách chọn số cụm `k`.
- `category_encoded` là mã số hóa danh mục, nên chỉ mang tính đại diện đơn giản cho dữ liệu phân loại.
- Chưa sử dụng thêm các đặc trưng như ngày bán, số lượt xem, giảm giá hoặc doanh thu.

## 8. Hướng phát triển

- Tăng kích thước dataset.
- Thử thêm các thuật toán khác như DBSCAN hoặc Hierarchical Clustering.
- Bổ sung thêm biểu đồ tương tác bằng Plotly.
- Thêm bộ lọc nâng cao trên dashboard.
