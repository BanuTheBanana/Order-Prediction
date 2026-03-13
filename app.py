import streamlit as st
import pandas as pd
import numpy as np

# Set trang hiển thị full màn hình cho giống giao diện web thật
st.set_page_config(page_title="Amazon Predictor", layout="wide")

# Tạo 3 Tabs như yêu cầu
tab1, tab2, tab3 = st.tabs(["🎯 Prediction Machine", "📊 Data Exploration", "💬 Project Assistant"])

# ----------------------------------------------------------------------
# TAB 1: PREDICTION MACHINE
# ----------------------------------------------------------------------
with tab1:
    # Thêm tiêu đề có màu sắc nổi bật và căn giữa
    st.markdown("<h2 style='text-align: center; color: #1E88E5;'>🎯 Phân Tích & Dự Đoán Hủy Đơn Hàng</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #666; font-size: 1.1rem;'>Dựa trên mô hình học máy (Random Forest) để dự báo khả năng khách hàng hủy đơn hàng E-commerce.</p>", unsafe_allow_html=True)
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
                price = st.number_input("💸 Giá trị đơn hàng (INR)", min_value=0.0, value=49133.0, step=100.0, format="%.2f")
            with c2:
                quantity = st.number_input("📦 Số lượng sản phẩm", min_value=1, value=1, step=1)
            
            st.divider()
            
            st.markdown("##### 2. Đặc trưng mã hóa (Encoded Features)")
            # Chia làm 2 cột nhỏ cho các tham số Model
            f_col1, f_col2 = st.columns(2)
            
            with f_col1:
                is_business = st.selectbox(
                    "🏢 Khách doanh nghiệp?", 
                    options=[0, 1], 
                    format_func=lambda x: "Có (1)" if x == 1 else "Không (0)"
                )
                shipping_type = st.selectbox("🚚 Hình thức vận chuyển", options=[0, 1])
                fulfilment_int = st.selectbox("🏭 Kênh hoàn thành", options=[0, 1])

            with f_col2:
                size = st.selectbox("📏 Kích cỡ (Size)", options=[0, 1, 2, 3, 4, 5])
                promotion_int = st.selectbox("🎁 Mức ưu đãi áp dụng", options=[0, 1, 2, 3])

            st.write("") # Tạo một chút khoảng trống thả lỏng không gian

        # Nút bấm mở rộng (Chiếm toàn bộ chiều ngang), dùng style chuẩn 
        if st.button("🚀 Chạy Mô Hình Dự Đoán", type="primary", use_container_width=True):
            
            # DataFrame được tạo ra từ input
            input_data = pd.DataFrame({
                'Quantity': [quantity],
                'Price': [price],
                'Is_Business': [is_business],
                'Size': [size],
                'Shipping_Type': [shipping_type],
                'Promotion_int': [promotion_int],
                'Fulfilment_Int': [fulfilment_int]
            })
            
            # Khối UI chạy loading... tạo cảm giác AI đang làm việc thật
            with st.spinner("Đang phân tích dữ liệu qua mô hình Random Forest..."):
                import time
                time.sleep(1.5) # Giả lập delay một chút cho ngầu
                
                # Mock kết quả trả về
                with st.container(border=True):
                    st.success("✅ Phân tích hoàn tất!")
                    res_col1, res_col2 = st.columns([1, 1.5])
                    
                    with res_col1:
                        # Dùng metric để giật tít kết quả giống Dashboard thực sự
                        st.metric(
                            label="Khả năng bị Hủy (Mock)", 
                            value="18%", 
                            delta="-5% so với TB", 
                            delta_color="inverse"
                        )
                        st.caption("Mô hình dự đoán: Rủi ro thấp. Bạn có thể chốt đơn an toàn.")

                    with res_col2:
                        st.info("Dữ liệu vector đầu vào (Vector Data):")
                        st.dataframe(input_data, hide_index=True)

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
            Dưới đây là các biến quan trọng được sử dụng trong mô hình Random Forest:
            
            - 📦 **Quantity**: Số lượng sản phẩm
            - 💸 **Price**: Giá trị giao dịch (INR)
            - 🏢 **Is_Business**: Khách hàng doanh nghiệp (1=Có, 0=Không)
            - 📏 **Size**: Kích thước sản phẩm (Mã hóa: 0-5)
            - 🚚 **Shipping_Type**: Phương thức vận chuyển
            - 🎁 **Promotion_int**: Mức độ khuyến mãi áp dụng
            - 🏭 **Fulfilment_Int**: Kênh hoàn thành đơn
            - ⏳ **Customer tenure**: Thời gian khách hàng gắn bó *(Extra)*
            - 💳 **Payment type**: Hình thức thanh toán *(Extra)*
            """)
            st.info("💡 **Ghi chú:** Các biến nhãn chuỗi (Text) đều đã được chuyển đổi (One-Hot / Label Encoding) để đưa vào mô hình máy học.")

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
    st.markdown("<p style='text-align: center; color: #666; font-size: 1.1rem;'>Hỏi đáp trực tiếp với trợ lý ảo về chi tiết dự án, Random Forest, hay Data Cleaning.</p>", unsafe_allow_html=True)
    st.divider()
    
    # Chia layout để phần chat ko bị tràn viền quá rộng (giống ChatGPT)
    _, chat_col, _ = st.columns([1, 4, 1])
    
    with chat_col:
        system_prompt = """
        You are an AI assistant for a data science project created by a second-year Artificial Intelligence student at FPT University. 
        The project predicts e-commerce order cancellation probabilities using a Random Forest model. 
        The dataset features include order amount, customer tenure, discount application, shipping method, payment type, and item count.
        
        Your rules:
        - Answer questions strictly related to this e-commerce data, Random Forest models, data cleaning (like One-Hot Encoding), and Exploratory Data Analysis.
        - If a user asks about unrelated topics (e.g., coding help, general history, weather), politely decline and state that you can only answer questions about the order cancellation project.
        - Keep answers concise, professional, and educational.
        - For now the content of the project is empty, so you can only tell them to wait for more information to be added.
        """

        # Thêm một info box hướng dẫn nhanh
        st.info("🤖 **Gợi ý câu hỏi:** *'Random Forest hoạt động như thế nào?'* hoặc *'Data Dictionary của dữ liệu này gồm gì?'*")

        # Container bao ngoài vùng tin nhắn để tạo khung viền đẹp
        with st.container(border=True, height=450): # Giới hạn chiều cao có croll
            # Khởi tạo history cho chat trong session state
            if "messages" not in st.session_state:
                st.session_state.messages = []
                st.session_state.messages.append({"role": "assistant", "content": "👋 Xin chào! Tôi là AI Assistant của dự án (sinh viên AI năm 2 ĐH FPT). Tôi có thể giúp gì cho bạn về Random Forest, Data Cleaning, hoặc mô hình phân tích hủy đơn hàng?"})

            # Hiển thị các tin nhắn trước đó
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

        # Input nằm ngay dưới container viền chat
        if prompt := st.chat_input("Gõ câu hỏi của bạn tại đây..."):
            
            # Thêm tin nhắn user vào lịch sử
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Hàm refresh mini để hiển thị ngay text bạn vừa gõ (Workaround mượt trên form streamlit)
            st.rerun()

        # Xử lý Logic AI (Nếu tin nhắn cuối cùng là của User -> AI cần trả lời)
        if len(st.session_state.messages) > 0 and st.session_state.messages[-1]["role"] == "user":
            with chat_col: # Vẫn render ngoài container chat viền để dùng chat_message mới sinh ra
                # Phải render lại tin nhắn user vừa gõ (do rerun đã reset UI nhưng ko render logic lặp bên trên nữa với tin nhắn mới nhất nếu k viết khéo, nhưng vì ta dùng rerun nên logic lặp ở vòng lặp for trên kia SẼ LÀM RỒI)
                # Tuy nhiên form input mới nằm bên ngoài vòng for -> đoạn logic genbot này cần ở ngoài
                pass
            
            # Giả lập suy nghĩ của bot (Bot typing...)
            with st.spinner("Đang suy luận..."):
                import time
                time.sleep(1)
                mock_response = "For now the content of the project is empty, so please wait for more information to be added. *(Please integrate with an LLM API to enable full capabilities)*"
                
                st.session_state.messages.append({"role": "assistant", "content": mock_response})
                st.rerun()