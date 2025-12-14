# Dự Án Nhận Diện Trái Cây và Vật Thể (AI Fruit Vision)

Ứng dụng web nhận diện vật thể thời gian thực sử dụng mạng nơ-ron tích chập (CNN) MobileNetV2 và thư viện xử lý ảnh OpenCV. Dự án được tối ưu hóa để triển khai trên môi trường cloud serverless như Vercel.

## 1. Giới Thiệu
Ứng dụng cho phép người dùng sử dụng camera của thiết bị (điện thoại, laptop) để nhận diện các loại trái cây và vật thể phổ biến. Hệ thống sẽ phân tích hình ảnh và vẽ khung bao quanh vật thể cùng với tên và độ tin cậy của dự đoán.

## 2. Mô Hình AI & Nguyên Lý Hoạt Động

### Mô hình sử dụng: **MobileNetV2**
Đây là một kiến trúc mạng CNN nhẹ (lightweight), được thiết kế đặc biệt cho các thiết bị di động và môi trường web với tài nguyên hạn chế.
- **Input**: Ảnh màu kích thước 128x128 pixel.
- **Output**: Xác suất thuộc về một trong 10 phân lớp.
- **Các phân lớp (Labels)**:
  1. Annona (Mãng cầu)
  2. Apples (Táo)
  3. Bananas (Chuối)
  4. Lemons (Chanh)
  5. Mango (Xoài)
  6. Oranges (Cam)
  7. Tomatoes (Cà chua)
  8. Human (Người)
  9. Pen (Bút bi)
  10. Phone (Điện thoại)

### Nguyên Lý Xử Lý (Pipeline)
Quy trình xử lý một khung hình (frame) diễn ra như sau:

1.  **Client-Side Capture (Phía Client)**:
    - Trình duyệt sử dụng `navigator.mediaDevices.getUserMedia` để lấy luồng video từ camera người dùng.
    - Chụp lại frame hình ảnh và gửi về server qua API `/process_frame`.

2.  **Preprocessing (Tiền xử lý)**:
    - Server nhận ảnh, resize về kích thước chuẩn `128x128` (dựa trên `config.INPUT_SHAPE`).
    - Chuẩn hóa dữ liệu pixel (đưa về khoảng -1 đến 1 hoặc 0 đến 1 tùy theo `preprocess_input` của MobileNetV2).

3.  **Inference (Dự đoán)**:
    - Ảnh đã xử lý được đưa qua model `mobilenet.h5`.
    - Model trả về mảng xác suất cho 10 class. Class có xác suất cao nhất (`np.argmax`) được chọn làm kết quả.

4.  **Post-processing (Hậu xử lý & Computer Vision)**:
    - Sử dụng **OpenCV** để xử lý ảnh gốc:
        - Chuyển sang ảnh xám (Grayscale).
        - Nhị phân hóa (Thresholding) để tách nền.
        - Tìm đường viền (Contours) để xác định vị trí vật thể.
    - Vẽ khung chữ nhật (Bounding Box) quanh vật thể lớn nhất.
    - Viết tên vật thể và độ chính xác lên ảnh.

5.  **Response**:
    - Ảnh kết quả được mã hóa lại thành JPEG và gửi trả về trình duyệt để hiển thị.

## 3. Cấu Trúc Dự Án
- **`app.py`**: Server Flask chính. Chứa API `/process_frame` để nhận và xử lý ảnh.
- **`core.py`**: Chứa class `ObjectDetector`. Nơi xử lý logic AI và OpenCV.
  - *Lưu ý*: Có bản vá `FixedDepthwiseConv2D` để khắc phục lỗi tương thích phiên bản Keras.
- **`config.py`**: File cấu hình (đường dẫn model, danh sách labels, màu sắc).
- **`templates/index.html`**: Giao diện người dùng (UI) hiện đại với hiệu ứng Glassmorphism.
- **`mobilenet/mobilenet.h5`**: File model đã được huấn luyện (Weight).

## 4. Hướng Dẫn Cài Đặt & Chạy Local

### Bước 1: Cài đặt thư viện
```bash
pip install -r requirements.txt
```

### Bước 2: Chạy Server
```bash
python app.py
```
*Server sẽ chạy tại `http://127.0.0.1:5000` (hoặc 5001 tùy cấu hình).*

### Bước 3: Sử dụng
Mở trình duyệt, truy cập địa chỉ trên và bấm nút **"Start Camera"**.

## 5. Triển Khai (Deployment) trên Vercel
Dự án đã có sẵn file cấu hình `vercel.json`.
1.  Push code lên **GitHub**.
2.  Tạo project mới trên **Vercel** -> Import repo GitHub vừa push.
3.  Vercel sẽ tự động build và deploy.
    - *Lưu ý*: Do giới hạn kích thước của Serverless Function (250MB), nếu gặp lỗi "Bundle Size Exceeded", cần cân nhắc sử dụng model nhỏ hơn hoặc chuyển sang `tflite-runtime`.
