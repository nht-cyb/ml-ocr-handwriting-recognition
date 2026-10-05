import streamlit as st
from PIL import Image
import os
from tool.predictor import Predictor
from tool.config import Cfg

def main():
    st.title("Ứng dụng Dự đoán Văn bản trên Hình ảnh Handwritten")
    
    # Tải cấu hình mô hình
    config = Cfg.load_config_from_name('vgg_transformer')
    config['weights'] ='D:\VWork\N4\ML\CK\source_code\Handwritten_OCR\checkpoint\model2.pth'
    detector = Predictor(config)
    
    # Giao diện tải lên hình ảnh
    uploaded_image = st.file_uploader("Tải lên hình ảnh", type=["jpg", "png", "jpeg"])
    
    if uploaded_image is not None:
        img = Image.open(uploaded_image)
        
        # Hiển thị hình ảnh
        st.image(img, caption='Hình ảnh đã tải lên', use_column_width=True)
        
        # Thực hiện dự đoán
        label = detector.predict(img)
        
        # Hiển thị nhãn dự đoán
        st.write(f'Nhãn dự đoán: {label}')

if __name__ == '__main__':
    main()


