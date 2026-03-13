import streamlit as st
import pandas as pdd
import numpy as np

# Set trang hiển thị full màn hình cho giống giao diện web thật
st.set_page_config(page_title="Amazon Predictor", layout="wide")

st.markdown("## ASUS ROG G700 (2025) Gaming Desktop PC - Dự đoán đơn hàng")
st.markdown("---")

# Chia layout giống Amazon: Cột trái (Ảnh), Cột phải (Thông số/Mua hàng)
col1, col2 = st.columns([1, 1.2])

with col1:
    # Chèn ảnh minh họa (Bạn có thể thay bằng đường dẫn ảnh thật của bạn)
    st.image("https://www.jordan1.vn/wp-content/uploads/2023/10/assc-kkoch-black-tee-1627659076.png", use_container_width=True)
    st.markdown("*Hình ảnh chỉ mang tính chất minh họa cho sản phẩm*")

with col2:
    st.subheader("Tùy chỉnh thông số đơn hàng")
    st.write("Vui lòng nhập các thông tin dưới đây để đưa vào Model dự đoán.")
    
    # 1. Price & Quantity (Float & Integer)
    price = st.number_input("Price (Transaction value in INR)", min_value=0.0, value=49133.0, step=100.0, format="%.2f")
    quantity = st.number_input("Quantity (Order quantity)", min_value=1, value=1, step=1)
    
    st.markdown("---")
    st.markdown("**Thông tin bổ sung (Encoded Features)**")
    
    # Chia làm 2 cột nhỏ cho form đỡ dài
    f_col1, f_col2 = st.columns(2)
    
    with f_col1:
        # Is Business (Binary)
        is_business = st.selectbox(
            "Is Business (Khách doanh nghiệp?)", 
            options=[0, 1], 
            format_func=lambda x: "True (1)" if x == 1 else "False (0)"
        )
        
        # Shipping_Type (Binary - Encoded)
        shipping_type = st.selectbox("Shipping Type", options=[0, 1])
        
        # Fulfilment_Int (Binary - Encoded)
        fulfilment_int = st.selectbox("Fulfilment Channel", options=[0, 1])

    with f_col2:
        # Size (Integer - Encoded) - Giả sử size có các label từ 0-5
        size = st.selectbox("Size (Garment size)", options=[0, 1, 2, 3, 4, 5])
        
        # Promotion_int (Integer - Encoded) - Giả sử có vài mức KM
        promotion_int = st.selectbox("Promotion Applied", options=[0, 1, 2, 3])

    st.markdown("<br>", unsafe_allow_html=True)

    # Nút bấm dự đoán (Giống nút Add to cart)
    if st.button("Dự đoán kết quả (Predict) 🛒", type="primary", use_container_width=True):
        
        # Tóm tắt dữ liệu thành DataFrame để đưa vào model
        input_data = pd.DataFrame({
            'Quantity': [quantity],
            'Price': [price],
            'Is_Business': [is_business],
            'Size': [size],
            'Shipping_Type': [shipping_type],
            'Promotion_int': [promotion_int],
            'Fulfilment_Int': [fulfilment_int]
        })
        
        with st.spinner("Đang chạy mô hình dự đoán..."):
            # CHỖ NÀY BẠN LOAD MODEL VÀO:
            # import joblib
            # model = joblib.load('model_cua_ban.pkl')
            # prediction = model.predict(input_data)
            
            # Khúc này tôi đang mock kết quả trả về
            st.success("✅ Dự đoán thành công!")
            st.info("Dữ liệu nhóm bạn chuẩn bị đưa vào model:")
            st.dataframe(input_data, hide_index=True)
            
            # st.write(f"### Kết quả: {prediction[0]}")