import streamlit as st
from ultralytics import YOLO
from PIL import Image
import os

# --- 1. CẤU HÌNH TRANG WEB ---
st.set_page_config(page_title="Human Detection App", layout="wide")

# --- 2. GIAO DIỆN TIÊU ĐỀ ---
st.title("Human Detection ")
st.write("Trương Công Thành - 223332852")
st.markdown("---")

# --- 3. TẢI MÔ HÌNH ---
@st.cache_resource
def load_model():
    # Kiểm tra nếu có file best.pt thì dùng, không thì dùng yolov8n.pt mặc định
    if os.path.exists("best.pt"):
        return YOLO("best.pt")
    else:
        st.warning("⚠️ Không tìm thấy 'best.pt', đang sử dụng mô hình mặc định yolov8n.pt")
        return YOLO("yolov8n.pt") 

model = load_model()

# --- 4. CẤU HÌNH ĐẦU VÀO ---
st.sidebar.header("Cấu hình")
input_type = st.sidebar.radio("Chọn nguồn ảnh:", ("Tải ảnh lên", "Sử dụng Webcam"))


source_img = None

if input_type == "Tải ảnh lên":
    uploaded_file = st.file_uploader("Chọn ảnh từ máy tính...", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        source_img = Image.open(uploaded_file)
else:
    cam_file = st.camera_input("Chụp ảnh từ Webcam")
    if cam_file:
        source_img = Image.open(cam_file)

# --- 5. XỬ LÝ VÀ HIỂN THỊ KẾT QUẢ ---
if source_img is not None:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Ảnh đầu vào")
        st.image(source_img, use_container_width=True)
    
    with st.spinner('Đang phân tích...'):
        # CHỈ NHẬN DIỆN NGƯỜI (classes=[0])
        results = model.predict(source=source_img, conf=0.5, classes=[0]) 
        res_plotted = results[0].plot()
        count = len(results[0].boxes)

    with col2:
        st.subheader("Kết quả nhận diện")
        st.image(res_plotted, use_container_width=True)
        
        if count > 0:
            st.success(f"✅ Xác nhận: Tìm thấy {count} người.")
        else:
            st.warning("⚠️ Không phát hiện thấy người.")

