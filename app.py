import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

# --- 1. CẤU HÌNH TRANG WEB ---
st.set_page_config(
    page_title="Human Detection App",
    page_icon="👦",
    layout="wide"
)

# --- 2. GIAO DIỆN TIÊU ĐỀ (GIỐNG ẢNH MẪU) ---
st.title("👦 Human Detection")
st.write("Truong Cong Thanh - 223332852")
st.write("Upload ảnh để phát hiện có phải người hay không.")

st.markdown("---")

# --- 3. TẢI MÔ HÌNH ---
# Hàm này giúp cache mô hình để không phải load lại mỗi khi bạn bấm nút
@st.cache_resource
def load_model():
    # Đảm bảo file best.pt nằm cùng thư mục với file app.py này
    return YOLO("best.pt")

try:
    model = load_model()
except Exception as e:
    st.error(f"Lỗi: Không tìm thấy file 'best.pt' trong thư mục. Vui lòng kiểm tra lại!")
    st.stop()

# --- 4. BỐ CỤC CHÍNH (GỒM 2 CỘT) ---
left_col, right_col = st.columns([1, 1])

with left_col:
    st.subheader("📁 Chọn ảnh từ máy tính")
    uploaded_file = st.file_uploader(
        "Drag and drop file here", 
        type=["jpg", "jpeg", "png"],
        help="Giới hạn 200MB mỗi file"
    )

with right_col:
    st.subheader("📊 Kết quả phân tích")
    # Khu vực này sẽ hiển thị kết quả sau khi xử lý

# --- 5. XỬ LÝ ẢNH VÀ HIỂN THỊ ---
if uploaded_file is not None:
    # Đọc ảnh từ file upload
    image = Image.open(uploaded_file)
    
    with left_col:
        st.image(image, caption="Ảnh gốc đã tải lên", use_container_width=True)
        btn_analyze = st.button("Nhấn để Submit và xem kết quả")

    if btn_analyze:
        with st.spinner('Đang nhận diện...'):
            # Chạy mô hình dự đoán
            results = model.predict(source=image, conf=0.25)
            
            # Vẽ kết quả lên ảnh
            res_plotted = results[0].plot()
            
            # Đếm số lượng người (Class 0 trong bộ COCO/Human là người)
            # Lưu ý: Nếu bạn train bộ dữ liệu chỉ có 1 lớp, class id luôn là 0
            count = len(results[0].boxes) 

        with right_col:
            # Hiển thị ảnh đã được vẽ khung nhận diện
            st.image(res_plotted, caption="Kết quả phát hiện", use_container_width=True)
            
            # Hiển thị thông báo số lượng
            if count > 0:
                st.success(f"Tìm thấy {count} người trong ảnh!")
            else:
                st.warning("Không tìm thấy người nào trong ảnh này.")
else:
    with right_col:
        st.info("Chọn ảnh và nhấn Submit để xem kết quả")

# --- 6. CHÂN TRANG ---
st.markdown("---")
st.caption("Ứng dụng được phát triển trên nền tảng Streamlit & YOLOv8/v11")