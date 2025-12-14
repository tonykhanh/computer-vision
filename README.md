# Dự Án Nhận Diện Trái Cây và Vật Thể (AI Fruit Vision) - Streamlit Version

Ứng dụng web nhận diện vật thể sử dụng mạng nơ-ron tích chập (CNN) MobileNetV2 và thư viện xử lý ảnh OpenCV. Dự án này đã được chuyển đổi sang **Streamlit** để đơn giản hóa quá trình phát triển và triển khai.

## 1. Giới Thiệu
Ứng dụng cho phép người dùng chụp ảnh từ camera của thiết bị (điện thoại, laptop) để nhận diện các loại trái cây và vật thể phổ biến. Mọi xử lý đều diễn ra trên server Python.

## 2. Mô Hình AI & Công Nghệ
### Mô hình: **MobileNetV2** (TensorFlow Lite / Keras)
- **Input**: Ảnh màu kích thước 128x128 pixel.
- **Output**: Xác suất thuộc về một trong 10 phân lớp.
- **Labels**: Mãng cầu, Táo, Chuối, Chanh, Xoài, Cam, Cà chua, Người, Bút bi, Điện thoại.

### Công nghệ: **Streamlit**
Streamlit giúp xây dựng giao diện web tương tác cho các dự án Data Science và AI một cách nhanh chóng mà không cần viết HTML/CSS/JS phức tạp.

## 3. Cấu Trúc Dự Án
- **`streamlit_app.py`**: File chính của ứng dụng. Quản lý giao diện và logic tương tác.
- **`core.py`**: Chứa class `ObjectDetector`. Nơi xử lý logic AI và OpenCV (Inference, Drawing).
- **`config.py`**: File cấu hình (đường dẫn model, danh sách labels, màu sắc).
- **`mobilenet/model.tflite`**: File model TFLite đã được tối ưu hóa.

## 4. Hướng Dẫn Cài Đặt & Chạy Local

### Bước 1: Cài đặt thư viện
Đảm bảo bạn đã cài đặt Python (khuyên dùng 3.9 - 3.12).
```bash
pip install -r requirements.txt
```

### Bước 2: Chạy Ứng dụng
```bash
streamlit run streamlit_app.py
```
*Trình duyệt sẽ tự động mở tại `http://localhost:8501`.*

### Bước 3: Sử dụng
1. Cấp quyền truy cập Camera cho trình duyệt.
2. Bấm nút "Take Photo" (hoặc chụp ảnh) trên giao diện.
3. Ứng dụng sẽ hiển thị ảnh đã qua xử lý với khung nhận diện.

## 5. Triển Khai (Deployment)

Cách tốt nhất để triển khai ứng dụng Streamlit là sử dụng **Streamlit Community Cloud** (miễn phí):

1.  Push code lên **GitHub** (public repository).
2.  Truy cập [share.streamlit.io](https://share.streamlit.io).
3.  Kết nối tài khoản GitHub.
4.  Chọn repo của bạn:
    -   **Main file path**: `streamlit_app.py`
5.  Bấm **Deploy**.
