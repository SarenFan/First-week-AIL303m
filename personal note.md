# 📘 Machine Learning & Data Science Notes

## 1. Quy trình làm việc của Machine Learning (Workflow)

1. **Xác định vấn đề**  
   - Hiểu rõ mục tiêu: dự đoán, phân loại, phân cụm, gợi ý…  
   - Ví dụ: Dự đoán giá nhà, phân loại email spam/không spam.

2. **Thu thập dữ liệu (Data Collection)**  
   - Nguồn dữ liệu: CSDL (SQL/NoSQL), file CSV, API, web scraping.  
   - Lưu ý: Kiểm tra chất lượng dữ liệu ngay từ đầu.

3. **Khám phá dữ liệu & Tiền xử lý (EDA & Pre-processing)**  
   - Kiểm tra kích thước, kiểu dữ liệu, giá trị null.  
   - Visualization (matplotlib, seaborn, plotly).  
   - Làm sạch dữ liệu (missing values, outliers).  
   - Feature engineering: tạo thêm biến có ý nghĩa.

4. **Xây dựng mô hình (Model Building)**  
   - Chọn mô hình: hồi quy tuyến tính, logistic, decision tree, random forest, neural network.  
   - Huấn luyện mô hình bằng dữ liệu train.  

5. **Đánh giá & Triển khai (Evaluation & Deployment)**  
   - Đánh giá bằng tập test và các metric phù hợp (Accuracy, RMSE, MAE, AUC, F1-score…).  
   - Triển khai mô hình vào hệ thống thật (API, Web app).

---

## 2. Làm việc với Dữ liệu trong Python

### 2.1 Pandas
- **Đọc CSV**:  
  ```python
  import pandas as pd
  df = pd.read_csv("data.csv")
  ```
- Chuyển đổi dữ liệu SQL query thành DataFrame:  
  ```python
  import sqlite3
  conn = sqlite3.connect("database.db")
  query = "SELECT * FROM users"
  df = pd.read_sql_query(query, conn)
  ```

### 2.2 SQL & Python
- Kết nối **sqlite3**:  
  ```python
  import sqlite3
  conn = sqlite3.connect("example.db")
  cursor = conn.cursor()
  cursor.execute("SELECT * FROM table_name")
  results = cursor.fetchall()
  ```

### 2.3 NoSQL & MongoDB
- **NoSQL**: CSDL phi quan hệ, dữ liệu thường dưới dạng **JSON**.  
  - Các loại phổ biến:  
    - Document DB (MongoDB)  
    - Graph DB (Neo4j)  
    - Wide-column DB (Cassandra)  
    - Key-Value DB (Redis)

- Kết nối MongoDB với Python (pymongo):  
  ```python
  from pymongo import MongoClient
  client = MongoClient("mongodb://localhost:27017/")
  db = client["mydatabase"]
  collection = db["users"]
  data = collection.find()
  ```

---

## 3. Xử lý Dữ liệu (Data Cleaning & Preprocessing)

### 3.1 Xử lý giá trị thiếu (Missing Values)
- **Xóa hàng có missing** → Nhanh nhưng có thể mất nhiều dữ liệu.  
- **Thay thế (imputation)**:  
  - Trung bình (mean), trung vị (median), mode.  
  - Sử dụng mô hình (KNN Imputer, Regression Imputer).

### 3.2 Biến đổi dữ liệu
- **Log Transformation**:  
  - Dùng cho dữ liệu lệch (skewed), ví dụ: giá nhà.  
  - `df["col"] = np.log1p(df["col"])`
- **One-hot Encoding**:  
  - `pd.get_dummies(df, columns=["category"])`
- **Chuẩn hóa (Scaling)**:  
  - MinMaxScaler: dữ liệu về [0,1]  
  - StandardScaler: dữ liệu có mean=0, std=1

### 3.3 Tạo đặc trưng mới (Feature Engineering)
- **Interaction Features**: kết hợp nhiều biến (nhân/chia).  
- Ví dụ: `df["price_per_area"] = df["price"] / df["area"]`

---

## 4. Thống kê & Suy diễn (Statistics & Inference)

- **Ước lượng (Estimation)**: đưa ra giá trị xấp xỉ cho tham số tổng thể.  
- **Sai số chuẩn (Standard Error)**: đo độ biến động của ước lượng.  
- **Suy diễn thống kê (Statistical Inference)**: từ mẫu → quần thể.  

### 4.1 Frequentist vs Bayesian
- **Frequentist**: chỉ dựa vào dữ liệu hiện có.  
- **Bayesian**: sử dụng cả dữ liệu hiện có + prior knowledge.  

Ví dụ:  
- Frequentist: "Xác suất tung xu ra ngửa là 0.5 (dựa vào nhiều lần tung)"  
- Bayesian: "Tôi nghĩ xu này hơi lệch, dựa vào kinh nghiệm trước, nên prior ≠ 0.5".

---

## 5. EDA (Exploratory Data Analysis)

- **Mục đích**: Hiểu dữ liệu trước khi model.  
- **Công cụ**:  
  - `df.describe()`  
  - `df.info()`  
  - Biểu đồ phân phối: hist, boxplot.  
  - Kiểm tra correlation: `df.corr()`

---

## 6. Machine Learning Models

### 6.1 Hồi quy tuyến tính (Linear Regression)
- Dễ dùng, dễ hiểu.  
- Nhược điểm: dễ overfit nếu dữ liệu phi tuyến.

### 6.2 Chính quy hóa (Regularization)
- **Ridge Regression (L2)**: giảm hệ số nhưng không loại bỏ.  
- **Lasso Regression (L1)**: có thể triệt tiêu hoàn toàn hệ số → chọn lọc đặc trưng.  
- **Elastic Net**: kết hợp cả hai.

### 6.3 Kiểm định chéo (Cross-validation)
- Dùng để chọn tham số tối ưu.  
- Ví dụ:  
  ```python
  from sklearn.linear_model import RidgeCV, LassoCV, ElasticNetCV
  model = RidgeCV(alphas=[0.1, 1, 10])
  model.fit(X_train, y_train)
  ```

---

# ✅ Tóm tắt
- Bắt đầu từ **hiểu vấn đề → thu thập dữ liệu → EDA → tiền xử lý → xây mô hình → đánh giá → triển khai**.  
- Sử dụng **Pandas** (CSV, SQL), **pymongo** (MongoDB).  
- Biết xử lý **missing values, outliers, skewed features**.  
- Phân biệt **Frequentist vs Bayesian**.  
- Thực hành **Linear Regression + Regularization** và **Cross-validation**.  
