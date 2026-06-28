# Tối ưu hóa Danh mục Đầu tư (Streamlit Web App)

Đây là một ứng dụng Web (Streamlit) được xây dựng dựa trên thuật toán tối ưu hóa danh mục đầu tư chứng khoán (HOSE 2020-2023) kết hợp chỉ báo RSI và MACD.

## Tính năng nổi bật
1. **Tiền xử lý và tạo tín hiệu**: Tự động tính toán các chỉ báo (RSI, MACD, Signal Line) và xây dựng lệnh Mua/Bán theo logic kỹ thuật tối ưu.
2. **Train model (2020)**: Tự động phân tích và chọn lọc Top 5 mã cổ phiếu tốt nhất từ danh sách dữ liệu có tỷ lệ Sharpe cao nhất.
3. **Kiểm định / Backtest (2021-2023)**:
    - Trực quan hóa lợi nhuận tích lũy của Chiến lược với biểu đồ Plotly tương tác.
    - So sánh với các danh mục Benchmark: Mua & Giữ Top 5, VN-INDEX, và 1/N toàn thị trường.
4. **Báo cáo chi tiết**:
    - Phân tích rủi ro & hiệu quả chi tiết (Return, Volatility, Sharpe, Max Drawdown,...).
    - Kiểm định Đa giai đoạn (Regime Testing) bao gồm Uptrend, Downtrend và Sideway.
    - Kiểm định Out-of-sample độc lập qua từng năm.

## Hướng dẫn cài đặt và chạy ứng dụng cục bộ (Local)

1. **Yêu cầu môi trường**: Python 3.8+
2. **Cài đặt các thư viện cần thiết**:
    ```bash
    pip install -r requirements.txt
    ```
3. **Chuẩn bị dữ liệu**: Đảm bảo tệp dữ liệu đầu vào `HOSE_2020_2023.csv` nằm ở cùng thư mục với `app.py`.
4. **Khởi chạy ứng dụng**:
    ```bash
    streamlit run app.py
    ```

## Triển khai (Deploy) lên Streamlit Community Cloud
Để đưa ứng dụng này lên mạng (public):
1. **Push lên GitHub**: Tải các file `app.py`, `requirements.txt`, `README.md` cùng file dữ liệu `HOSE_2020_2023.csv` lên 1 public repository trên GitHub của bạn. (Lưu ý: Nếu tệp CSV quá lớn, bạn có thể cân nhắc lưu trữ file ở nguồn khác rồi dùng Pandas để tải qua HTTP url).
2. **Triển khai**: 
    - Truy cập [Streamlit Share](https://share.streamlit.io/).
    - Đăng nhập và nhấp vào "New app".
    - Liên kết với Repository GitHub mà bạn vừa tạo và điền file chính (Main file path) là `app.py`.
    - Nhấp "Deploy" và chờ trong ít phút!
