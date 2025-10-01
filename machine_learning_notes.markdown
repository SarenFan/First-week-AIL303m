# 📘 Machine Learning & Data Science Notes

## 1. Quy trình làm việc của Machine Learning (Workflow)

![Quy trình Machine Learning](https://raw.githubusercontent.com/yourusername/yourrepo/main/images/ml_workflow.png)  
*Hình 1: Sơ đồ quy trình Machine Learning, từ xác định vấn đề đến giám sát mô hình.*

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

4. **Chia dữ liệu (Data Splitting)**  
   - Chia dữ liệu thành tập huấn luyện (train, 70-80%), tập kiểm tra (test, 20-30%), và tập xác thực (validation, nếu cần) để đánh giá mô hình khách quan, tránh overfitting.  
   - **Stratified Split**: Đối với dữ liệu không cân bằng (ví dụ: lớp thiểu số chỉ 10%), đảm bảo tỷ lệ lớp đồng đều ở các tập.  
   - Ví dụ mã Python với scikit-learn:  
     ```python
     from sklearn.model_selection import train_test_split
     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
     ```

5. **Xây dựng mô hình (Model Building)**  
   - Chọn mô hình: hồi quy tuyến tính, logistic, decision tree, random forest, neural network.  
   - Huấn luyện mô hình bằng dữ liệu train.  

6. **Điều chỉnh siêu tham số (Hyperparameter Tuning)**  
   - Tối ưu hóa siêu tham số (như alpha trong regularization) để cải thiện hiệu suất.  
   - Phương pháp:  
     - **Grid Search**: Thử tất cả tổ hợp siêu tham số trong lưới.  
     - **Random Search**: Thử ngẫu nhiên để tiết kiệm thời gian.  
     - **Bayesian Optimization**: Dùng xác suất để ưu tiên các giá trị tốt.  
   - Ví dụ Grid Search:  
     ```python
     from sklearn.model_selection import GridSearchCV
     from sklearn.linear_model import Ridge
     param_grid = {'alpha': [0.1, 1, 10]}
     grid_search = GridSearchCV(Ridge(), param_grid, cv=5)
     grid_search.fit(X_train, y_train)
     best_alpha = grid_search.best_params_['alpha']
     ```

7. **Đánh giá & Triển khai (Evaluation & Deployment)**  
   - Đánh giá bằng tập test với các metric (Accuracy, RMSE, MAE, AUC, F1-score…).  
   - Triển khai mô hình vào hệ thống thực tế (API, Web app).  
   - **Giám sát và bảo trì**: Theo dõi hiệu suất (data drift), huấn luyện lại định kỳ, A/B testing để so sánh mô hình mới/cũ.  
   - Công cụ: MLflow, TensorBoard.

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
![Cấu trúc dữ liệu JSON](https://raw.githubusercontent.com/yourusername/yourrepo/main/images/json_structure.png)  
*Hình 2: So sánh dữ liệu JSON (MongoDB) với bảng SQL.*

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

### 2.4 Lấy dữ liệu từ API
- Sử dụng thư viện **requests** để lấy dữ liệu từ API (ví dụ: Twitter, Google Maps).  
- Xử lý authentication với API keys hoặc OAuth.  
- Ví dụ:  
  ```python
  import requests
  response = requests.get("https://api.example.com/data?api_key=your_key")
  data = response.json()
  df = pd.DataFrame(data['results'])
  ```

### 2.5 Xử lý dữ liệu lớn (Big Data Handling)
![So sánh hiệu suất](https://raw.githubusercontent.com/yourusername/yourrepo/main/images/pandas_dask_spark.png)  
*Hình 3: So sánh tốc độ xử lý của Pandas, Dask, và PySpark với dữ liệu lớn.*

- Với dữ liệu lớn (hàng triệu hàng), Pandas có thể chậm hoặc hết RAM.  
- **Dask**: Mở rộng Pandas, tính toán song song.  
- **PySpark**: Xử lý dữ liệu phân tán trên cluster (Hadoop, cloud).  
- Ví dụ với Dask:  
  ```python
  import dask.dataframe as dd
  df = dd.read_csv("large_data.csv")
  df = df.compute()  # Chỉ tính toán khi cần
  ```

---

## 3. Xử lý Dữ liệu (Data Cleaning & Preprocessing)

### 3.1 Xử lý giá trị thiếu (Missing Values)
![Heatmap giá trị null](https://raw.githubusercontent.com/yourusername/yourrepo/main/images/missing_values_heatmap.png)  
*Hình 4: Heatmap hiển thị giá trị null trước và sau khi xử lý.*

- **Xóa hàng có missing**: Nhanh nhưng có thể mất dữ liệu.  
- **Thay thế (imputation)**:  
  - Trung bình (mean), trung vị (median), mode.  
  - Sử dụng mô hình (KNN Imputer, Regression Imputer).  

### 3.2 Biến đổi dữ liệu
![Log Transformation](https://raw.githubusercontent.com/yourusername/yourrepo/main/images/log_transformation.png)  
*Hình 5: Phân phối dữ liệu trước và sau log transformation.*

- **Log Transformation**: Dùng cho dữ liệu lệch (skewed), ví dụ: giá nhà.  
  ```python
  import numpy as np
  df["col"] = np.log1p(df["col"])
  ```
- **One-hot Encoding**:  
  ```python
  pd.get_dummies(df, columns=["category"])
  ```
- **Chuẩn hóa (Scaling)**:  
  - MinMaxScaler: dữ liệu về [0,1].  
  - StandardScaler: mean=0, std=1.

### 3.3 Tạo đặc trưng mới (Feature Engineering)
- **Interaction Features**: Kết hợp nhiều biến (nhân/chia).  
  - Ví dụ: `df["price_per_area"] = df["price"] / df["area"]`

### 3.4 Xử lý dữ liệu không cân bằng (Imbalanced Data)
![SMOTE](https://raw.githubusercontent.com/yourusername/yourrepo/main/images/smote_before_after.png)  
*Hình 6: Minh họa dữ liệu trước và sau khi áp dụng SMOTE.*

- Trong phân loại, lớp thiểu số (ví dụ: 10% spam) gây thiên vị.  
- Phương pháp:  
  - **Oversampling**: Tăng dữ liệu lớp thiểu số (SMOTE).  
  - **Undersampling**: Giảm dữ liệu lớp đa số.  
  - **Class Weights**: Gán trọng số cao hơn cho lớp thiểu số.  
- Ví dụ SMOTE:  
  ```python
  from imblearn.over_sampling import SMOTE
  smote = SMOTE(random_state=42)
  X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
  ```

### 3.5 Xử lý dữ liệu văn bản và thời gian
- **Text Data**: Chuyển văn bản thành số (TF-IDF, Word2Vec).  
  ```python
  from sklearn.feature_extraction.text import TfidfVectorizer
  vectorizer = TfidfVectorizer()
  X_text = vectorizer.fit_transform(df["text_column"])
  ```
- **Datetime**: Trích xuất đặc trưng (ngày, tháng, giờ).  
  ```python
  df['date'] = pd.to_datetime(df['date'])
  df['month'] = df['date'].dt.month
  ```

---

## 4. Thống kê & Suy diễn (Statistics & Inference)

- **Ước lượng (Estimation)**: Đưa ra giá trị xấp xỉ cho tham số tổng thể.  
- **Sai số chuẩn (Standard Error)**: Đo độ biến động của ước lượng.  
- **Suy diễn thống kê (Statistical Inference)**: Từ mẫu suy ra quần thể.

### 4.1 Frequentist vs Bayesian
- **Frequentist**: Chỉ dựa vào dữ liệu hiện có.  
- **Bayesian**: Kết hợp dữ liệu với prior knowledge.  
- Ví dụ:  
  - Frequentist: "Xác suất tung xu ra ngửa là 0.5 (dựa vào nhiều lần tung)."  
  - Bayesian: "Dựa trên kinh nghiệm, xu hơi lệch, prior ≠ 0.5."

### 4.2 Kiểm định giả thuyết (Hypothesis Testing)
![p-value](https://raw.githubusercontent.com/yourusername/yourrepo/main/images/p_value_distribution.png)  
*Hình 7: Minh họa p-value và vùng bác bỏ trong kiểm định giả thuyết.*

- **Null Hypothesis (H0)**: Giả thuyết mặc định (không có sự khác biệt).  
- **Alternative Hypothesis (H1)**: Giả thuyết thay thế.  
- **p-value**: Nếu p < 0.05, bác bỏ H0.  
- Ví dụ t-test:  
  ```python
  from scipy.stats import ttest_ind
  stat, p_value = ttest_ind(group1, group2)
  if p_value < 0.05:
      print("Có sự khác biệt đáng kể")
  ```

### 4.3 Bias-Variance Tradeoff
![Bias-Variance Tradeoff](https://raw.githubusercontent.com/yourusername/yourrepo/main/images/bias_variance_tradeoff.png)  
*Hình 8: Đồ thị minh họa mối quan hệ giữa bias, variance và lỗi tổng quát.*

- **Bias Cao (Underfitting)**: Mô hình quá đơn giản, không nắm bắt pattern.  
- **Variance Cao (Overfitting)**: Mô hình quá phức tạp, học nhiễu.  
- **Tradeoff**: Cân bằng bằng regularization hoặc cross-validation.

---

## 5. EDA (Exploratory Data Analysis)

![Correlation Heatmap](https://raw.githubusercontent.com/yourusername/yourrepo/main/images/correlation_heatmap.png)  
*Hình 9: Heatmap hiển thị correlation giữa các biến.*

- **Mục đích**: Hiểu dữ liệu trước khi xây mô hình.  
- **Công cụ**:  
  - `df.describe()`  
  - `df.info()`  
  - Biểu đồ phân phối: histogram, boxplot.  
  - Kiểm tra correlation: `df.corr()`

![Boxplot](https://raw.githubusercontent.com/yourusername/yourrepo/main/images/boxplot_distribution.png)  
*Hình 10: Boxplot minh họa phân phối và outliers của một biến.*

### 5.1 Phát hiện đa cộng tuyến (Multicollinearity Detection)
![VIF Plot](https://raw.githubusercontent.com/yourusername/yourrepo/main/images/vif_plot.png)  
*Hình 11: Biểu đồ VIF cho các biến độc lập.*

- Biến độc lập tương quan cao gây bất ổn cho mô hình.  
- **VIF (Variance Inflation Factor)**: VIF > 5 thì loại bỏ biến.  
  ```python
  from statsmodels.stats.outliers_influence import variance_inflation_factor
  vif_data = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
  ```

### 5.2 Lựa chọn đặc trưng (Feature Selection)
- Giảm số biến để tăng tốc độ, giảm overfitting.  
- **Filter Methods**: Dựa trên thống kê (correlation, chi-square).  
- **Wrapper Methods**: Thử nghiệm với mô hình (forward/backward selection).  
- **Embedded Methods**: Trong mô hình (Lasso tự loại biến).

---

## 6. Machine Learning Models

### 6.1 Hồi quy tuyến tính (Linear Regression)
- Dễ dùng, dễ hiểu.  
- Nhược điểm: Dễ overfit nếu dữ liệu phi tuyến.

### 6.2 Chính quy hóa (Regularization)
- **Ridge Regression (L2)**: Giảm hệ số nhưng không loại bỏ.  
- **Lasso Regression (L1)**: Triệt tiêu hệ số, chọn lọc đặc trưng.  
- **Elastic Net**: Kết hợp L1 và L2.  

### 6.3 Kiểm định chéo (Cross-validation)
- Chọn tham số tối ưu, tránh overfitting.  
  ```python
  from sklearn.linear_model import RidgeCV
  model = RidgeCV(alphas=[0.1, 1, 10])
  model.fit(X_train, y_train)
  ```

### 6.4 Mô hình phân loại (Classification Models)
![Decision Tree](https://raw.githubusercontent.com/yourusername/yourrepo/main/images/decision_tree.png)  
*Hình 12: Cấu trúc của một decision tree.*

![SVM Hyperplane](https://raw.githubusercontent.com/yourusername/yourrepo/main/images/svm_hyperplane.png)  
*Hình 13: Minh họa hyperplane trong SVM.*

- **Logistic Regression**: Phân loại nhị phân (sigmoid).  
- **Decision Trees**: Xử lý phi tuyến, dễ overfit.  
- **Random Forest**: Kết hợp nhiều trees, giảm variance.  
- **SVM**: Tìm hyperplane phân cách tối ưu.  
  ```python
  from sklearn.linear_model import LogisticRegression
  model = LogisticRegression()
  model.fit(X_train, y_train)
  ```

### 6.5 Học không giám sát (Unsupervised Learning)
- **Clustering (K-Means)**: Nhóm dữ liệu tương tự, chọn K bằng elbow method.  
- **PCA**: Giảm chiều dữ liệu, giữ thông tin chính.  
  ```python
  from sklearn.cluster import KMeans
  kmeans = KMeans(n_clusters=3)
  clusters = kmeans.fit_predict(X)
  ```

### 6.6 Neural Networks
![Neural Network](https://raw.githubusercontent.com/yourusername/yourrepo/main/images/neural_network.png)  
*Hình 14: Sơ đồ cấu trúc neural network với input, hidden layers, và output.*

- Mô hình phức tạp cho dữ liệu lớn (hình ảnh, văn bản).  
- **Cấu trúc**: Input, hidden layers, output với activation (ReLU, sigmoid).  
  ```python
  from keras.models import Sequential
  from keras.layers import Dense
  model = Sequential()
  model.add(Dense(10, activation='relu', input_shape=(n_features,)))
  model.add(Dense(1, activation='sigmoid'))
  ```

---

## 7. Đạo đức và Thực tiễn trong ML (Ethics & Best Practices)

![Model Bias](https://raw.githubusercontent.com/yourusername/yourrepo/main/images/model_bias.png)  
*Hình 15: Minh họa bias trong dự đoán của mô hình.*

- **Bias và Fairness**: Mô hình có thể thiên vị (ví dụ: phân biệt giới tính trong tuyển dụng). Giải pháp: Kiểm tra dữ liệu, dùng fairness metrics (demographic parity).  
- **Privacy**: Bảo vệ dữ liệu cá nhân (GDPR), dùng anonymization hoặc differential privacy.  
- **Explainability**: Giải thích mô hình "hộp đen" với SHAP hoặc LIME.  

![SHAP Summary](https://raw.githubusercontent.com/yourusername/yourrepo/main/images/shap_summary.png)  
*Hình 16: SHAP summary plot giải thích tầm quan trọng của các biến.*

```python
import shap
explainer = shap.Explainer(model)
shap_values = explainer(X_test)
shap.summary_plot(shap_values, X_test)
```

---

## 8. Hiển thị ảnh trên GitHub

Khi upload file Markdown lên GitHub, đôi khi ảnh không hiển thị do lỗi đường dẫn hoặc cấu hình. Dưới đây là các bước để đảm bảo ảnh hiển thị đúng:

### 8.1 Kiểm tra upload ảnh
- Đảm bảo thư mục `images/` chứa tất cả ảnh (như `ml_workflow.png`, `json_structure.png`, v.v.) đã được commit và push lên GitHub.
- Truy cập repo trên GitHub, kiểm tra xem thư mục `images/` có xuất hiện với các file ảnh không.
- Lệnh Git:
  ```bash
  git add images/
  git commit -m "Add images for Markdown"
  git push origin main
  ```

### 8.2 Sử dụng đường dẫn đúng
- **Đường dẫn tương đối**: Nếu file Markdown ở root repo, dùng:
  ```markdown
  ![Tên hình](images/filename.png)
  ```
  - Đảm bảo tên file đúng (phân biệt chữ hoa/thường, không dấu cách, dùng `_` hoặc `-`).
- **Raw URL (khuyến nghị)**: Lấy URL raw từ GitHub:
  1. Mở ảnh trên GitHub (ví dụ: `https://github.com/yourusername/yourrepo/blob/main/images/ml_workflow.png`).
  2. Click nút **Raw** để lấy URL: `https://raw.githubusercontent.com/yourusername/yourrepo/main/images/ml_workflow.png`.
  3. Dùng trong Markdown:
     ```markdown
     ![Tên hình](https://raw.githubusercontent.com/yourusername/yourrepo/main/images/filename.png)
     ```
- Thay `yourusername/yourrepo` bằng tên người dùng và repo thực tế của bạn.

### 8.3 Kiểm tra lỗi
- **Case-sensitive**: Đảm bảo tên file chính xác (ví dụ: `Image.png` ≠ `image.png`).
- **Cache**: Refresh trang GitHub (Ctrl+F5) hoặc xóa cache browser.
- **Mobile**: Nếu ảnh không load trên app GitHub, thử trên desktop hoặc dùng VPN (một số ISP chặn `raw.githubusercontent.com`).
- **Developer Tools**: Mở F12 > Console/Network, reload trang để xem lỗi (như 404 nếu đường dẫn sai).

### 8.4 Khắc phục lỗi 404 Not Found
Nếu click vào link raw (như `https://raw.githubusercontent.com/yourusername/yourrepo/main/images/ml_workflow.png`) và nhận lỗi **404 Not Found**, hãy làm như sau:
- **Kiểm tra thư mục `images/`**: 
  - Mở repo trên GitHub, đảm bảo thư mục `images/` tồn tại và chứa file ảnh đúng tên (ví dụ: `ml_workflow.png`).
  - Nếu không thấy: Tạo thư mục `images/` cục bộ, thêm ảnh, và commit:
    ```bash
    mkdir images
    # Copy ảnh vào images/
    git add images/
    git commit -m "Add images for Markdown"
    git push origin main
    ```
- **Kiểm tra tên file**: Đảm bảo tên file không có dấu cách (dùng `_` thay vì khoảng trắng, ví dụ: `model_bias.png` thay vì `model bias.png`) và phân biệt chữ hoa/thường.
- **Kiểm tra branch**: Nếu repo dùng `master` thay vì `main`, sửa URL thành:
  ```markdown
  ![Tên hình](https://raw.githubusercontent.com/yourusername/yourrepo/master/images/filename.png)
  ```
- **Tạo ảnh nếu chưa có**: Dùng Python để tạo (ví dụ heatmap):
  ```python
  import seaborn as sns
  import matplotlib.pyplot as plt
  import pandas as pd
  import numpy as np
  df = pd.DataFrame(np.random.rand(10, 5), columns=['A', 'B', 'C', 'D', 'E'])
  plt.figure(figsize=(8, 6))
  sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
  plt.title('Correlation Heatmap')
  plt.savefig('images/correlation_heatmap.png', dpi=300, bbox_inches='tight')
  plt.show()
  ```
- **Test link raw**: Copy URL raw của ảnh, mở trong tab mới. Nếu vẫn 404, kiểm tra lại bước commit.

### 8.5 GitHub Pages (nếu sử dụng)
- Đặt ảnh trong thư mục `static/` hoặc `assets/`.
- Cập nhật đường dẫn trong Markdown, ví dụ:
  ```markdown
  ![Tên hình](/assets/filename.png)
  ```
- Kiểm tra file `_config.yml` để đảm bảo cấu hình đúng.

### 8.6 Debug nếu vẫn lỗi
- Cung cấp link repo GitHub (ví dụ: `https://github.com/username/repo`).
- Kiểm tra tên file ảnh cụ thể (ví dụ: `ml_workflow.png`).
- Mở F12 > Console/Network trên trình duyệt để xem lỗi chi tiết (như 404, 403).

---

# ✅ Tóm tắt
- Quy trình ML: Hiểu vấn đề → Thu thập dữ liệu → EDA → Tiền xử lý → Chia dữ liệu → Xây mô hình → Điều chỉnh siêu tham số → Đánh giá → Triển khai → Giám sát.  
- Công cụ: **Pandas** (CSV, SQL), **pymongo** (MongoDB), **Dask/PySpark** (big data), **requests** (API).  
- Xử lý dữ liệu: Missing values, outliers, imbalanced data, text/datetime.  
- Thống kê: Hypothesis testing, bias-variance tradeoff.  
- EDA: Multicollinearity, feature selection.  
- Mô hình: Hồi quy, phân loại, không giám sát, neural networks.  
- Thực hành ML có đạo đức: Bias, privacy, explainability.  
- **GitHub**: Đảm bảo ảnh hiển thị bằng cách commit thư mục `images/`, dùng raw URL, và khắc phục lỗi 404.