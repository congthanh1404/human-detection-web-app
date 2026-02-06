import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

# --- 1. CẤU HÌNH TRANG WEB ---
st.set_page_config(page_title="Human Detection App", page_icon="👦", layout="wide")

# --- 2. GIAO DIỆN TIÊU ĐỀ ---
st.title(" Human Detection ")
st.write("Trương Công Thành - 223332852")
st.markdown("---")

# --- 3. LOAD MÔ HÌNH ---
@st.cache_resource
def load_model():
    # Sử dụng mô hình bạn đã huấn luyện hoặc mô hình gốc yolov8n.pt
    return YOLO("best.pt") 

try:
    model = load_model()
except Exception as e:
    st.error("Không tìm thấy file mô hình 'best.pt'. Vui lòng kiểm tra trên GitHub!")
    st.stop()

# --- 4. CHỌN PHƯƠNG THỨC ĐẦU VÀO ---
st.sidebar.header("Cấu hình")
input_type = st.sidebar.radio("Chọn nguồn ảnh:", ("Tải ảnh lên", "Sử dụng Webcam"))
conf_threshold = st.sidebar.slider("Ngưỡng tin cậy (Confidence)", 0.0, 1.0, 0.5)

source_img = None

if input_type == "Tải ảnh lên":
    uploaded_file = st.file_uploader("Chọn ảnh từ máy tính...", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        source_img = Image.open(uploaded_file)
else:
    cam_file = st.camera_input("Chụp ảnh từ Webcam để kiểm tra")
    if cam_file:
        source_img = Image.open(cam_file)

# --- 5. XỬ LÝ VÀ HIỂN THỊ KẾT QUẢ ---
if source_img is not None:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Ảnh đầu vào")
        st.image(source_img, use_container_width=True)
    
    # Tự động chạy nhận diện khi có ảnh
    with st.spinner('Đang phân tích...'):
        # QUAN TRỌNG: classes=[0] để chỉ nhận diện người, conf lọc bỏ nhận diện yếu
        results = model.predict(source=source_img, conf=conf_threshold, classes=[0]) 
        
        # Vẽ khung kết quả (chỉ có khung người)
        res_plotted = results[0].plot()
        # Đếm số lượng người thực tế
        count = len(results[0].boxes)

    with col2:
        st.subheader("Kết quả nhận diện")
        st.image(res_plotted, use_container_width=True)
        
        if count > 0:
            st.success(f"✅ Xác nhận: Tìm thấy {count} người trong ảnh.")
        else:
            st.warning("⚠️ Không phát hiện thấy người.")

# --- 6. CHÂN TRANG ---
st.markdown("---")
st.caption("Mô hình đã được cấu hình để bỏ qua các vật dụng như ghế, đồng hồ, quạt...")
