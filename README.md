# 🧠 Customer Segmentation using RFM Analysis and Clustering

## 🎯 Business Objective / Problem

Cửa hàng X là một siêu thị bán lẻ chuyên cung cấp các sản phẩm thiết yếu như:
- Rau, củ, quả
- Thịt, cá, trứng
- Sữa, nước giải khát, thực phẩm đóng gói,...

Tệp khách hàng chính là **người tiêu dùng cá nhân**, thường xuyên ghé mua hàng hằng ngày hoặc hàng tuần.

Tuy nhiên, chủ cửa hàng hiện đang gặp phải một số vấn đề trong hoạt động kinh doanh:

- ❓ **Không biết nhóm khách hàng nào mua hàng thường xuyên nhất.**
- ❓ **Chưa rõ nên giới thiệu sản phẩm mới đến đối tượng nào để hiệu quả cao.**
- ❓ **Khó khăn trong việc xây dựng chương trình khuyến mãi phù hợp cho từng nhóm khách hàng.**
- ❓ **Chưa có chiến lược chăm sóc khách hàng cá nhân hóa.**

Do đó, chủ cửa hàng mong muốn **phân loại khách hàng** dựa trên hành vi tiêu dùng nhằm:
- 🎯 Tăng doanh thu thông qua tiếp thị đúng đối tượng.
- 🤝 Giữ chân khách hàng thân thiết bằng chính sách chăm sóc phù hợp.
- 📊 Có cơ sở dữ liệu để đưa ra quyết định kinh doanh tốt hơn.

Giải pháp được đề xuất là áp dụng **phân tích RFM kết hợp với mô hình phân cụm** (KMeans, GMM) để phân nhóm khách hàng và hỗ trợ ra quyết định marketing.


## 📌 Mục tiêu đề tài
Dự án nhằm **phân khúc khách hàng** dựa trên hành vi mua sắm, giúp doanh nghiệp cá nhân hoá chiến dịch marketing và tối ưu chiến lược kinh doanh. Phương pháp sử dụng:
- **RFM Analysis**: Phân tích tần suất và giá trị giao dịch.
- **KMeans & GMM Clustering**: Gom nhóm khách hàng theo điểm số RFM.

---

## 📁 Cấu trúc dự án

```bash
.
├── app.py                         # Ứng dụng chính sử dụng Streamlit
├── requirements.txt              # Danh sách thư viện cần cài
├── Procfile                      # Khởi chạy app trên Streamlit Cloud
├── setup.sh                      # Shell script hỗ trợ cài đặt môi trường
│
├── data/                         # Thư mục chứa dữ liệu
│   ├── Products_with_Categories.csv  # Danh sách sản phẩm, giá và phân loại
│   ├── Transactions.csv              # Lịch sử giao dịch của khách hàng
│   ├── rfm_data.pkl                  # File đã tính toán xong RFM và gom cụm
│   ├── sample_rfm_input.csv         # Dữ liệu đầu vào mẫu cho demo
│
├── models/                       # Thư mục chứa mô hình huấn luyện
│   ├── kmeans_model.pkl              # Mô hình KMeans đã huấn luyện
│   ├── rfm_scaler.pkl                # Bộ scaler chuẩn hoá RFM
│
├── images/                       # Hình ảnh minh hoạ và đồ thị
│   ├── *.png                         # Biểu đồ, trực quan hoá EDA, phân cụm
│   ├── *.jpg                         # Hình minh hoạ Customer Segmentation
```
## 🚀 Cách chạy ứng dụng

Sau khi đã cài đặt đầy đủ các thư viện cần thiết trong `requirements.txt`, bạn có thể khởi chạy ứng dụng bằng Streamlit với lệnh sau trong terminal:

```bash
streamlit run app.py
```

# 🔀 Các bước triển khai theo quy trình Data Science

## 📌 Bước 1: Business Understanding
- **Xác định vấn đề kinh doanh**: cải thiện quảng bá, tăng doanh thu, chăm sóc khách hàng.
- **Mục tiêu**: phân cụm khách hàng từ dữ liệu giao dịch, từ đó xây dựng chiến lược tiếp cận từng nhóm khách hàng hiệu quả hơn.

## 📌 Bước 2: Data Understanding / Acquire
- Dữ liệu đầu vào gồm 2 file CSV:
  - `Transactions.csv`: Giao dịch của khách hàng, gồm các cột: Member_number, Date, productId, items
  - `Products_with_Categories.csv`: Danh sách sản phẩm kèm theo giá và danh mục sản phẩm

## 📊 Dữ liệu sử dụng

### 1. `Transactions.csv`
Gồm các giao dịch mua sắm của khách hàng:

| Member_number | Date       | productId | items |
|---------------|------------|-----------|-------|
| 1808          | 21-07-2015 | 1         | 3     |
| 2552          | 05-01-2015 | 2         | 1     |
| ...           | ...        | ...       | ...   |

### 2. `Products_with_Categories.csv`
Thông tin sản phẩm, giá và danh mục:

| productId | productName        | price | Category          |
|-----------|--------------------|-------|-------------------|
| 1         | tropical fruit      | 7.8   | Fresh Food        |
| 2         | whole milk          | 1.8   | Dairy             |
| 3         | pip fruit           | 3.0   | Fresh Food        |
| ...       | ...                | ...   | ...               |

---

## 📌 Bước 3: Data Preparation / Prepare
- Làm sạch và xử lý dữ liệu giao dịch
- Tính toán các giá trị **R (Recency)**, **F (Frequency)**, **M (Monetary)** cho từng khách hàng

Ảnh minh họa phân phối Data Preparation:

![Data Preparation](https://github.com/GiangSon-5/gui_kmeans/blob/main/images/Data%20preparation.jpg)

## 📌 Bước 4 & 5: Modeling & Evaluation
- **Phân cụm khách hàng dựa trên RFM**:
  - RFM + KMeans
  - RFM + Hierarchical Clustering
- **So sánh, đánh giá hiệu quả từng mô hình** bằng trực quan hóa (bubble chart, histogram, PCA...) 🔍

## 📌 Bước 6: Deployment & Feedback / Act
- **Ứng dụng mô hình** trong chiến dịch marketing, ưu đãi và chăm sóc khách hàng theo từng phân nhóm.
- **Triển khai giao diện tương tác với người dùng** bằng Streamlit.




## 🔍 Phân tích RFM

- **Recency**: Thời gian kể từ lần mua hàng gần nhất.
- **Frequency**: Tần suất mua hàng.
- **Monetary**: Tổng chi tiêu.

Ảnh minh họa phân phối RFM:

![RFM](https://github.com/GiangSon-5/gui_kmeans/blob/main/images/RFM.png)

---

## 🔄 Gom cụm khách hàng

Sau khi chuẩn hoá RFM, mô hình **KMeans** và **GMM** được áp dụng để phân cụm. Mỗi khách hàng được gán vào 1 nhóm theo hành vi mua hàng.

Ảnh minh họa phân cụm:

![Customer Segmentation](https://github.com/GiangSon-5/gui_kmeans/blob/main/images/Customer-Segmentation.jpg)

---

## 🛠️ Công nghệ sử dụng

- Python (Pandas, Scikit-learn, Matplotlib, Seaborn)
- Streamlit (Giao diện người dùng)
- Joblib (Lưu mô hình)
- Git + GitHub (Quản lý phiên bản)
- ...
---

## 🚀 Hướng phát triển
- Kết hợp thêm dữ liệu thời gian thực
- Thử nghiệm mô hình phân cụm nâng cao (DBSCAN, HDBSCAN)
- Xây dựng tính năng gợi ý sản phẩm dựa trên phân khúc
