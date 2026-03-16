import streamlit as st
import pandas as pd
import numpy as np
import joblib
import lightgbm as lgb
# Set trang hiển thị full màn hình cho giống giao diện web thật
st.set_page_config(page_title="Amazon Predictor", layout="wide")

# ==========================================
# TẢI MÔ HÌNH (Sử dụng cache để tối ưu)
# ==========================================
@st.cache_resource
def load_model():
    # Điền đúng tên file joblib đã lưu cùng thư mục với script này
    return joblib.load("LG_model.joblib")
try:
    model = load_model()
except Exception as e:
    st.error(f"⚠️ Không thể tải mô hình: {e}. Vui lòng kiểm tra lại đường dẫn file 'LG_model.joblib'")
    st.stop()

# ==========================================
# GIAO DIỆN CHÍNH
# ==========================================
# Tạo 3 Tabs như yêu cầu
tab1, tab2, tab3 = st.tabs(["🎯 Prediction Machine", "📊 Data Exploration", "💬 Project Assistant"])

# ----------------------------------------------------------------------
# TAB 1: PREDICTION MACHINE
# ----------------------------------------------------------------------
with tab1:
    # Thêm tiêu đề có màu sắc nổi bật và căn giữa
    st.markdown("<h2 style='text-align: center; color: #1E88E5;'>🎯 Phân Tích & Dự Đoán Hủy Đơn Hàng</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #666; font-size: 1.1rem;'>Dựa trên mô hình học máy (LightGBM) để dự báo khả năng khách hàng hủy đơn hàng E-commerce.</p>", unsafe_allow_html=True)
    st.divider()

    # Chia layout: Cột trái (Ảnh & Demo), Cột phải (Form điền thông số) - Có khoảng cách (col_space)
    col1, col_space, col2 = st.columns([1, 0.1, 1.3])

    with col1:
        # Nhóm thông tin sản phẩm bằng st.container để bo góc đẹp hơn
        with st.container(border=True):
            st.markdown("#### 💻 Sản phẩm đang xem")
            # Chèn ảnh minh họa
            st.image("https://www.jordan1.vn/wp-content/uploads/2023/10/assc-kkoch-black-tee-1627659076.png", use_container_width=True)
            st.markdown("**ASUS ROG G700 (2025) Gaming Desktop PC**")
            st.caption("*Hình ảnh mang tính chất minh họa cho sản phẩm trong giỏ hàng.*")
            
            # Thông tin thêm về chính sách mua hàng ảo
            st.info("📦 **Giao hàng miễn phí** cho đơn hàng trên 50,000 INR.\n\n"
                    "🔄 **Đổi trả 30 ngày** theo chính sách của Amazon.")

    with col2:
        with st.container(border=True):
            st.subheader("⚙️ Tùy chỉnh thông số đầu vào")
            st.write("Vui lòng nhập các thông tin của đơn hàng để hệ thống đánh giá.")
            
            st.markdown("##### 1. Thông tin cơ bản")
            # Dùng 2 cột ngang cho Price và Quantity để tiết kiệm diện tích và nhìn đối xứng
            c1, c2 = st.columns(2)
            with c1:
                amount = st.number_input("💸 Giá trị đơn hàng (INR)", min_value=0.0, value=49133.0, step=100.0, format="%.2f")
            with c2:
                quantity = st.number_input("📦 Số lượng sản phẩm", min_value=1, value=1, step=1)
            
            st.divider()
            
            st.markdown("##### 2. Đặc trưng mã hóa (Encoded Features)")
            # Chia làm 2 cột nhỏ cho các tham số Model
            f_col1, f_col2 = st.columns(2)
            
            with f_col1:
                B2B_binary = st.selectbox(
                    "🏢 Khách doanh nghiệp?", 
                    options=[0, 1], 
                    format_func=lambda x: "Có (1)" if x == 1 else "Không (0)"
                )
                fulfillment_binary = st.selectbox("🏭 Kênh hoàn thành", options=[0, 1])

            with f_col2:
                size_ordinal = st.selectbox("📏 Kích cỡ (Size)", options=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
                promotion = st.selectbox("🎁 Có ưu đãi áp dụng", options=[0, 1])

            st.write("") # Tạo một chút khoảng trống thả lỏng không gian

        # Nút bấm mở rộng (Chiếm toàn bộ chiều ngang), dùng style chuẩn 
        if st.button("🚀 Chạy Mô Hình Dự Đoán", type="primary", use_container_width=True):
            
            # Trình tự các cột phải KHỚP 100% với trình tự lúc train model
            # Sắp xếp lại trình tự đặc trưng theo yêu cầu
            input_data = pd.DataFrame({
                'Qty': [quantity],
                'Amount': [amount],
                'fulfillment_binary': [fulfillment_binary],
                'promotion': [promotion],
                'size_ordinal': [size_ordinal],
                'B2B_binary': [B2B_binary]
            })            
            # Khối UI chạy loading... tạo cảm giác AI đang làm việc thật
            with st.spinner("Đang chạy mô hình AI để dự đoán khả năng hủy đơn..."):
                import time
                time.sleep(1.2) # Giả lập delay một chút
                
                # ------ DỰ ĐOÁN THỰC TẾ TỪ MODEL ------
                try:
                    # Lấy xác suất của class 1 (Khả năng hủy đơn)
                    cancel_prob = model.predict_proba(input_data)[0][1]
                    cancel_prob_pct = cancel_prob * 100
                    
                    # Logic hiển thị theo độ rủi ro
                    if cancel_prob > 0.5:
                        risk_text = "Rủi ro cao. Khách hàng có khả năng hủy đơn."
                        delta_text = "Cảnh báo cao"
                        delta_color = "inverse" # Đỏ
                    elif cancel_prob > 0.3:
                        risk_text = "Rủi ro trung bình. Cần theo dõi thêm."
                        delta_text = "Chú ý"
                        delta_color = "off"     # Xám
                    else:
                        risk_text = "Rủi ro thấp. Bạn có thể chốt đơn an toàn."
                        delta_text = "An toàn"
                        delta_color = "normal"  # Xanh
                    
                    with st.container(border=True):
                        st.success("✅ Phân tích hoàn tất!")
                        res_col1, res_col2 = st.columns([1, 1.5])
                        
                        with res_col1:
                            st.metric(
                                label="Khả năng bị Hủy", 
                                value=f"{cancel_prob_pct:.1f}%", 
                                delta=delta_text, 
                                delta_color=delta_color
                            )
                            st.caption(f"**Kết luận AI:** {risk_text}")

                        with res_col2:
                            st.info("Dữ liệu vector đầu vào (Đã đưa vào mô hình):")
                            st.dataframe(input_data, hide_index=True)
                            
                except Exception as e:
                    st.error(f"Lỗi trong quá trình dự đoán: {e}")

# ----------------------------------------------------------------------
# TAB 2: EXPLORATORY DATA ANALYSIS (EDA)
# ----------------------------------------------------------------------
with tab2:
    st.markdown("<h2 style='text-align: center; color: #4CAF50;'>📊 Khám Phá Dữ Liệu (EDA)</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #666; font-size: 1.1rem;'>Hiểu rõ các đặc trưng (features) ảnh hưởng thế nào đến quyết định hủy đơn hàng.</p>", unsafe_allow_html=True)
    st.divider()
    
    # Chia bố cục: Cột trái (Giải thích Features), Cột phải (Mock Chart)
    eda_col1, eda_col_space, eda_col2 = st.columns([1, 0.1, 1.5])
    
    with eda_col1:
        with st.container(border=True):
            st.subheader("📚 Từ Điển Dữ Liệu")
            st.markdown("""
            Dưới đây là các biến quan trọng được sử dụng trong mô hình LightGBM:
            
            - 📦 **Quantity**: Số lượng sản phẩm
            - 💸 **Price**: Giá trị giao dịch (INR)
            - 🏢 **Is_Business**: Khách hàng doanh nghiệp (1=Có, 0=Không)
            - 📏 **Size**: Kích thước sản phẩm (Mã hóa: 0-5)
            - 🚚 **Shipping_Type**: Phương thức vận chuyển
            - 🎁 **Promotion_int**: Mức độ khuyến mãi áp dụng
            - 🏭 **Fulfilment_Int**: Kênh hoàn thành đơn
            """)
            st.info("💡 **Ghi chú:** Các biến nhãn chuỗi (Text) đều đã được chuyển đổi để đưa vào mô hình máy học.")

    with eda_col2:
        with st.container(border=True):
            st.subheader("📈 Phân tích tương quan (Mock Data)")
            
            # Hiển thị 3 metrics tổng quan ảo
            m1, m2, m3 = st.columns(3)
            m1.metric("Tỉ lệ hủy đơn TB", "12.4%", "-2.1% (tháng này)", delta_color="inverse")
            m2.metric("Đơn KH Doanh nghiệp", "45%", "+5% (tháng này)")
            m3.metric("Áp dụng Khuyến mãi", "68%", "Ổn định", delta_color="off")
            
            st.write("")
            st.markdown("##### 📉 Biểu đồ phân phối giá trị theo tỉ lệ hủy (Minh họa)")
            # Vẽ một cái Bar Chart bằng st.bar_chart với data random ảo
            chart_data = pd.DataFrame(
                np.random.randint(10, 100, size=(6, 2)),
                columns=["Không hủy", "Hủy đơn"],
                index=["Size 0", "Size 1", "Size 2", "Size 3", "Size 4", "Size 5"]
            )
            st.bar_chart(chart_data, color=["#1E88E5", "#FF4B4B"], use_container_width=True)

# ----------------------------------------------------------------------
# TAB 3: PROJECT ASSISTANT (CHATBOT)
# ----------------------------------------------------------------------
with tab3:
    st.markdown("<h2 style='text-align: center; color: #8E24AA;'>💬 Trợ Lý AI Nội Bộ</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #666; font-size: 1.1rem;'>Hỏi đáp trực tiếp với trợ lý ảo về chi tiết dự án, Machine Learning, hay Data Cleaning.</p>", unsafe_allow_html=True)
    st.divider()
    
    # Chia layout để phần chat ko bị tràn viền quá rộng
    _, chat_col, _ = st.columns([1, 4, 1])
    
    with chat_col:
        st.info("🤖 **Gợi ý câu hỏi:** *'Mô hình hoạt động như thế nào?'* hoặc *'Data Dictionary của dữ liệu này gồm gì?'*")

        with st.container(border=True, height=450):
            if "messages" not in st.session_state:
                st.session_state.messages = []
                st.session_state.messages.append({"role": "assistant", "content": "👋 Xin chào! Tôi là AI Assistant của dự án (sinh viên AI năm 2 ĐH FPT). Tôi có thể giúp gì cho bạn về mô hình phân tích hủy đơn hàng?"})

            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

        if prompt := st.chat_input("Gõ câu hỏi của bạn tại đây..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.rerun()

        if len(st.session_state.messages) > 0 and st.session_state.messages[-1]["role"] == "user":
            with st.spinner("Đang suy luận..."):
                import time
                time.sleep(1)
                mock_response = "Giao diện Chatbot hiện chưa được tích hợp LLM. Vui lòng kết nối API (như OpenAI) để trợ lý này có thể trò chuyện thật."
                st.session_state.messages.append({"role": "assistant", "content": mock_response})
                st.rerun()
