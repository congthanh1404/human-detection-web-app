import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

# --- 1. CẤU HÌNH TRANG WEB ---
st.set_page_config(page_title="Human Detection App", page_icon="👦", layout="wide")

# --- 2. GIAO DIỆN TIÊU ĐỀ ---
st.title(" Human Detection ")
st.write("Trương Công Thành - 223332852")

# --- 3. LOAD MÔ HÌNH ---
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

# --- 4. CHỌN PHƯƠNG THỨC ĐẦU VÀO ---
st.sidebar.header("Cấu hình đầu vào")
input_type = st.sidebar.radio("Chọn nguồn ảnh:", ("Tải ảnh lên", "Sử dụng Webcam"))

# Biến chứa dữ liệu ảnh
source_img = None

if input_type == "Tải ảnh lên":
    uploaded_file = st.file_uploader("Chọn ảnh từ máy tính...", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        source_img = Image.open(uploaded_file)
else:
    # Chức năng chụp ảnh từ Webcam
    cam_file = st.camera_input("Chụp ảnh để nhận diện người")
    if cam_file:
        source_img = Image.open(cam_file)

# --- 5. XỬ LÝ VÀ HIỂN THỊ KẾT QUẢ ---
if source_img is not None:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Ảnh đầu vào")
        st.image(source_img, use_container_width=True)
    
    # Nút bấm kích hoạt nhận diện
    if st.button("Bắt đầu nhận diện"):
        with st.spinner('Đang phân tích...'):
            results = model.predict(source_img, conf=0.25)
            res_plotted = results[0].plot()
            count = len(results[0].boxes)

        with col2:
            st.subheader("Kết quả")
            st.image(res_plotted, use_container_width=True)
            if count > 0:
                st.success(f"Tìm thấy {count} người!")
            else:
                st.warning("Không tìm thấy người.")
