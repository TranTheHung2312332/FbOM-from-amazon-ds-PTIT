# Amazon Review ABSA Mining Project

## Giới thiệu

Đây là bài tập lớn phục vụ cho môn học **Khai phá dữ liệu lớn**, sử dụng **Amazon Review Dataset 2023** để nghiên cứu và xây dựng hệ thống phân tích quan điểm người dùng trên dữ liệu đánh giá sản phẩm.

Dự án tập trung vào bài toán **Aspect-Based Sentiment Analysis (ABSA)**. Khác với phân tích cảm xúc thông thường chỉ xác định cảm xúc chung của toàn bộ review, ABSA đi sâu hơn bằng cách xác định:

- Người dùng đang nhắc đến **khía cạnh nào** của sản phẩm.
- Cảm xúc của người dùng đối với **từng khía cạnh cụ thể** là tích cực, tiêu cực hay trung lập.

Ví dụ:

```text
Sentence:
The battery life is excellent but the screen is dim.

Output:
battery life -> positive
screen -> negative
```

Pipeline tổng quát của dự án gồm hai bài toán chính:

```text
ATE -> ASC
```

Trong đó:

- **ATE — Aspect Term Extraction**: trích xuất các aspect hoặc khía cạnh sản phẩm được nhắc đến trong câu.
- **ASC — Aspect Sentiment Classification**: phân loại sentiment tương ứng với từng aspect.

---

## Mục tiêu dự án

Dự án hướng đến các mục tiêu chính:

- Xử lý và khai phá dữ liệu review Amazon ở quy mô lớn.
- Tiền xử lý văn bản review và chuyển dữ liệu về dạng phù hợp cho bài toán ABSA.
- Trích xuất các aspect quan trọng được người dùng nhắc đến trong đánh giá sản phẩm.
- Phân loại sentiment cho từng aspect theo ba nhãn:

```text
0 = negative
1 = neutral
2 = positive
```

- Xây dựng pipeline có khả năng mở rộng trên dữ liệu lớn.
- Đánh giá mô hình bằng các metric phù hợp với dữ liệu mất cân bằng, đặc biệt là macro-F1 và per-class F1.

---

## Dữ liệu

Nguồn dữ liệu chính:

```text
Amazon Review Dataset 2023
```

Các category được sử dụng trong dự án:

- Electronics
- Software
- Office Products
- Kindle Store

Dữ liệu review Amazon có kích thước lớn, nhiều nhiễu và phân bố không đồng đều giữa các category. Vì vậy, dự án cần thực hiện các bước làm sạch, chuẩn hóa, lọc câu và tổ chức dữ liệu trước khi huấn luyện mô hình.

Dữ liệu sau xử lý được lưu chủ yếu ở định dạng:

```text
Parquet
CSV
```

---

## Bài toán chính

### Aspect Term Extraction

**Aspect Term Extraction (ATE)** là bài toán trích xuất các cụm từ thể hiện khía cạnh sản phẩm trong câu review.

Ví dụ:

```text
Sentence:
The battery life is excellent but the screen is dim.

Aspects:
battery life
screen
```

ATE giúp xác định người dùng đang nói về phần nào của sản phẩm, chẳng hạn như pin, màn hình, giá cả, chất lượng, hiệu năng hoặc dịch vụ.

---

### Aspect Sentiment Classification

**Aspect Sentiment Classification (ASC)** là bài toán phân loại sentiment cho từng aspect đã được xác định.

Ví dụ:

```text
Sentence:
The battery life is excellent but the screen is dim.

Aspect:
screen

Sentiment:
negative
```

ASC cần xử lý các trường hợp một câu có nhiều aspect với sentiment khác nhau. Vì vậy, mô hình không chỉ cần hiểu nội dung câu mà còn phải biết aspect nào đang được xét.

Một dạng input được sử dụng trong dự án là **aspect-marker format**:

```text
The battery life is excellent but the [ASP] screen [/ASP] is dim.
```

Cách biểu diễn này giúp mô hình nhận biết rõ aspect cần phân loại sentiment.

---

## Tổng quan phương pháp

Dự án xây dựng pipeline ABSA theo hướng kết hợp giữa dữ liệu gán nhãn thủ công và dữ liệu pseudo-label.

Các bước tổng quát gồm:

1. Tiền xử lý dữ liệu review Amazon.
2. Lọc và chọn các câu có khả năng chứa thông tin aspect/opinion.
3. Tạo tập dữ liệu gán nhãn nhỏ cho bài toán ABSA.
4. Huấn luyện mô hình trích xuất aspect.
5. Huấn luyện mô hình phân loại sentiment theo aspect.
6. Tạo pseudo-label trên tập dữ liệu lớn.
7. Kết hợp dữ liệu gold và pseudo-label để cải thiện mô hình.
8. Đánh giá mô hình trên tập test được gán nhãn thủ công.

---

## Định dạng dữ liệu

Một sample cho bài toán ASC thường có dạng:

```text
parent_asin
sentence_id
sentence_text
rating
aspect
sentiment
category_name
```

Trong đó:

- `parent_asin`: mã sản phẩm.
- `sentence_id`: mã định danh câu.
- `sentence_text`: câu review.
- `rating`: rating gốc của review.
- `aspect`: khía cạnh sản phẩm được nhắc đến.
- `sentiment`: nhãn hoặc phân phối sentiment.
- `category_name`: category của sản phẩm.

Ví dụ:

```text
sentence_text:
The battery life is excellent but the screen is dim.

aspect:
battery life

sentiment:
positive
```

---

## Mô hình sử dụng

Dự án sử dụng các mô hình học máy và học sâu cho xử lý ngôn ngữ tự nhiên, trong đó trọng tâm là các mô hình Transformer.

Một số hướng tiếp cận chính:

- Feature-based baseline.
- Token classification cho ATE.
- Transformer-based classifier cho ASC.
- RoBERTa-based sentiment classification.
- Self-training với pseudo-label.

Mô hình chính cho ASC sử dụng:

```text
roberta-base
```

với bài toán phân loại 3 lớp:

```text
negative / neutral / positive
```

---

## Đánh giá

Các metric chính được sử dụng:

- Accuracy
- Precision
- Recall
- F1-score
- Macro-F1
- Weighted-F1
- Per-class F1
- Confusion matrix

Do dữ liệu sentiment trong review thường bị mất cân bằng, đặc biệt là lớp neutral, dự án ưu tiên đánh giá bằng:

```text
macro-F1
per-class F1
confusion matrix
```

thay vì chỉ dựa vào accuracy.

---

## Công nghệ sử dụng

Các công nghệ và thư viện chính:

- Python
- PySpark
- Pandas
- Polars
- NumPy
- scikit-learn
- PyTorch
- Hugging Face Transformers
- Sentence Transformers
- Parquet
- Google Colab
- Google Drive

---

## Cấu trúc đầu ra chính

Một số dạng file đầu ra trong dự án:

```text
*.parquet
*.csv
*.txt
```

Ví dụ:

```text
high_confidence_samples_{category}.parquet
low_confidence_samples_{category}.parquet
train.csv
gold_test.csv
gold_test_metrics.csv
gold_test_classification_report.csv
gold_test_confusion_matrix.csv
```

---

## Trạng thái dự án

Dự án hiện tập trung vào việc hoàn thiện pipeline ABSA cho dữ liệu Amazon Review Dataset, bao gồm:

- Xử lý dữ liệu lớn.
- Trích xuất aspect.
- Phân loại sentiment theo aspect.
- Tạo và sử dụng pseudo-label.
- Huấn luyện mô hình trên dữ liệu kết hợp.
- Đánh giá mô hình trên gold test.

---

## Nhóm thực hiện

Dự án được thực hiện bởi nhóm sinh viên trong khuôn khổ môn học.

---

## License

Dự án sử dụng **MIT License**.