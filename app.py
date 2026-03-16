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
# Điều hướng "tab" tự động (Streamlit tabs không cho chuyển bằng code)
PAGES = [
    "🧪 Playground",
    "📊 Data Exploration",
    "💬 Project Assistant",
    "🛍️ Browser Sản Phẩm",
]
if "active_page" not in st.session_state:
    st.session_state.active_page = "🛍️ Browser Sản Phẩm"

# Xử lý điều hướng tự động (phải làm TRƯỚC khi widget radio được tạo)
if "nav_to_page" in st.session_state:
    st.session_state.active_page = st.session_state.pop("nav_to_page")
if "nav_to_subtab" in st.session_state:
    st.session_state.browser_subtab = st.session_state.pop("nav_to_subtab")

active_page = st.radio(
    "Điều hướng",
    PAGES,
    index=PAGES.index(st.session_state.active_page),
    horizontal=True,
    label_visibility="collapsed",
    key="active_page",
)


def run_prediction_ui(*, amount: float, locked_amount: bool, widget_prefix: str):
    col_left, col_right = st.columns([1, 1.3])

    with col_right:
        with st.container(border=True):
            st.subheader("⚙️ Tùy chỉnh thông số đầu vào")
            st.write("Điền thông tin để hệ thống đánh giá khả năng hủy đơn.")

            st.markdown("##### 1. Thông tin cơ bản")
            c1, c2 = st.columns(2)
            with c1:
                st.number_input(
                    "💸 Giá trị đơn hàng (INR)",
                    min_value=0.0,
                    value=float(amount),
                    step=100.0,
                    format="%.2f",
                    key=f"{widget_prefix}_amount",
                    disabled=locked_amount,
                )
            with c2:
                quantity = st.number_input(
                    "📦 Số lượng sản phẩm",
                    min_value=1,
                    value=1,
                    step=1,
                    key=f"{widget_prefix}_qty",
                )

            st.divider()
            st.markdown("##### 2. Đặc trưng mã hóa (Encoded Features)")
            f_col1, f_col2 = st.columns(2)

            with f_col1:
                B2B_binary = st.selectbox(
                    "🏢 Khách doanh nghiệp?",
                    options=[0, 1],
                    format_func=lambda x: "Có" if x == 1 else "Không",
                    key=f"{widget_prefix}_b2b",
                )
                fulfillment_binary = st.selectbox(
                    "🏭 Gói dịch vụ",
                    options=[0, 1],
                    format_func=lambda x: "Premium" if x == 1 else "Thường",
                    key=f"{widget_prefix}_fulfill",
                )

            with f_col2:
                size_labels = [
                    "Free",
                    "XS",
                    "S",
                    "M",
                    "L",
                    "XL",
                    "2XL",
                    "3XL",
                    "4XL",
                    "5XL",
                    "6XL",
                ]
                size_ordinal = st.selectbox(
                    "📏 Kích cỡ (Size)",
                    options=list(range(len(size_labels))),
                    format_func=lambda x: size_labels[int(x)],
                    key=f"{widget_prefix}_size",
                )
                promotion = st.selectbox(
                    "🎁 Ưu đãi",
                    options=[0, 1],
                    format_func=lambda x: "Có" if x == 1 else "Không",
                    key=f"{widget_prefix}_promo",
                )

        if st.button("🚀 Chạy Mô Hình Dự Đoán", type="primary", use_container_width=True, key=f"{widget_prefix}_run"):
            input_data = pd.DataFrame(
                {
                    "Qty": [quantity],
                    "Amount": [float(amount)],
                    "fulfillment_binary": [fulfillment_binary],
                    "promotion": [promotion],
                    "size_ordinal": [size_ordinal],
                    "B2B_binary": [B2B_binary],
                }
            )

            with st.spinner("Đang chạy mô hình AI để dự đoán khả năng thành công..."):
                import time

                time.sleep(1.0)
                try:
                    success_prob = float(model.predict_proba(input_data)[0][1])
                    success_prob_pct = success_prob * 100

                    if success_prob < 0.5:
                        risk_text = "Khả năng thành công thấp. Cần cân nhắc/kiểm tra thêm điều kiện đơn hàng."
                        delta_text = "Thấp"
                        delta_color = "inverse"
                    elif success_prob < 0.7:
                        risk_text = "Khả năng thành công trung bình. Nên theo dõi thêm."
                        delta_text = "Trung bình"
                        delta_color = "off"
                    else:
                        risk_text = "Khả năng thành công cao. Có thể chốt đơn an toàn."
                        delta_text = "Cao"
                        delta_color = "normal"

                    with st.container(border=True):
                        st.info("✅ Phân tích hoàn tất!")
                        res_col1, res_col2 = st.columns([1, 1.5])

                        with res_col1:
                            st.metric(
                                label="Khả năng thành công",
                                value=f"{success_prob_pct:.1f}%",
                                delta=delta_text,
                                delta_color=delta_color,
                            )
                            if success_prob < 0.5:
                                st.error(f"**Kết luận AI:** {risk_text}")
                            elif success_prob < 0.7:
                                st.warning(f"**Kết luận AI:** {risk_text}")
                            else:
                                st.success(f"**Kết luận AI:** {risk_text}")

                        with res_col2:
                            st.markdown("**Dữ liệu vector đầu vào (Đã đưa vào mô hình):**")
                            st.dataframe(input_data, hide_index=True)
                except Exception as e:
                    st.error(f"Lỗi trong quá trình dự đoán: {e}")

    with col_left:
        with st.container(border=True):
            st.markdown("#### 🧾 Thông tin phân tích")
            if locked_amount:
                st.caption("Giá đang được khóa theo sản phẩm đã chọn.")
            else:
                st.caption("Playground: bạn có thể đổi giá để thử các kịch bản.")


if active_page == "🧪 Playground":
    st.markdown("<h2 style='text-align: center; color: #1E88E5;'>🧪 Playground</h2>", unsafe_allow_html=True)
    st.markdown(
        "<p style='text-align: center; color: #666; font-size: 1.0rem;'>"
        "Khu vực thử nghiệm mô hình (không gắn với sản phẩm cụ thể)."
        "</p>",
        unsafe_allow_html=True,
    )
    st.divider()
    run_prediction_ui(amount=49133.0, locked_amount=False, widget_prefix="playground")

# ----------------------------------------------------------------------
# TAB 2: EXPLORATORY DATA ANALYSIS (EDA)
# ----------------------------------------------------------------------
elif active_page == "📊 Data Exploration":
    st.markdown("<h2 style='text-align: center; color: #4CAF50;'>📊 Khám Phá Dữ Liệu (EDA)</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #666; font-size: 1.1rem;'>Hiểu rõ các đặc trưng (features) ảnh hưởng thế nào đến quyết định hủy đơn hàng.</p>", unsafe_allow_html=True)
    st.divider()
    
    # Chia bố cục: Cột trái (Giải thích Features), Cột phải (Mock Chart)
    eda_col1, eda_col_space, eda_col2 = st.columns([1, 0.1, 1.5])
    
    with eda_col1:
        with st.container(border=True):
            st.subheader("📚 Từ Điển Dữ Liệu")
            st.markdown("""
            Dưới đây là các biến (feature) quan trọng trong mô hình LightGBM:

            - 📦 **Qty**: Số lượng sản phẩm đặt mua
            - 💸 **Amount**: Giá trị đơn hàng (INR)
            - 🏢 **B2B**: Khách hàng doanh nghiệp (1 = Có, 0 = Không)
            - 📏 **Size_Int**: Mã hóa kích cỡ sản phẩm (0: Nhỏ nhất, lớn dần đến 10)
            - 🚚 **Service_Level_Int**: Mã hóa loại dịch vụ giao hàng (0, 1, 2)
            - 🎁 **Promotion_Count**: Số lượng khuyến mãi được áp dụng
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
elif active_page == "💬 Project Assistant":
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

# ----------------------------------------------------------------------
# TAB 4: BROWSE SẢN PHẨM (CLOTHING) + TAB PHỤ
# ----------------------------------------------------------------------
elif active_page == "🛍️ Browser Sản Phẩm":
    st.markdown(
        "<h2 style='text-align: center; color: #F57C00;'>🛍️ Duyệt Sản Phẩm Thời Trang</h2>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='text-align: center; color: #666; font-size: 1.0rem;'>"
        "Demo nhanh giao diện duyệt quần áo (mock data, chưa kết nối chức năng)."
        "</p>",
        unsafe_allow_html=True,
    )
    st.divider()

    if "browser_subtab" not in st.session_state:
        st.session_state.browser_subtab = "🛒 Duyệt sản phẩm"

    browser_subtab = st.radio(
        "Tab phụ",
        ["🛒 Duyệt sản phẩm", "🎯 Dự đoán sản phẩm"],
        index=["🛒 Duyệt sản phẩm", "🎯 Dự đoán sản phẩm"].index(st.session_state.browser_subtab),
        horizontal=True,
        label_visibility="collapsed",
        key="browser_subtab",
    )

    selected_clothing_type = "Tất cả quần áo"

    # ------------------- KHU VỰC KẾT QUẢ / DỰ ĐOÁN -------------------
    selected_product = st.session_state.get("selected_product")

    # Thanh tìm kiếm (giữ lại, bỏ bộ lọc khác)
    with st.container(border=True):
        search_query = st.text_input(
            "Tìm kiếm quần áo",
            placeholder="Ví dụ: hoodie đen, áo thun oversize...",
        )

    st.write("")

    # Nếu đang ở subtab dự đoán thì render khu vực dự đoán và dừng
    if browser_subtab == "🎯 Dự đoán sản phẩm":
        if not selected_product:
            st.warning("Bạn chưa chọn sản phẩm. Hãy quay lại tab 🛒 Duyệt sản phẩm và bấm “Xem dự đoán”.")
        else:
            nav1, nav2, _ = st.columns([1, 1, 2])
            with nav1:
                if st.button("← Quay lại duyệt", use_container_width=True, key="back_to_browse"):
                    st.session_state["nav_to_subtab"] = "🛒 Duyệt sản phẩm"
                    st.rerun()
            with nav2:
                if st.button("Hủy chọn sản phẩm", use_container_width=True, key="cancel_product"):
                    st.session_state.pop("selected_product", None)
                    st.session_state["nav_to_subtab"] = "🛒 Duyệt sản phẩm"
                    st.rerun()

            with st.container(border=True):
                st.markdown("#### ✅ Sản phẩm đã chọn")
                p = selected_product
                pcol1, pcol2 = st.columns([1, 1.6])
                with pcol1:
                    st.image(p["image_url"], use_container_width=True)
                with pcol2:
                    st.markdown(f"**{p['name']}**")
                    st.caption(p["subtitle"])
                    st.markdown(
                        f"<span style='color:#B12704; font-weight:700; font-size: 1.05rem;'>₹ {p['price_inr']:,} INR</span>",
                        unsafe_allow_html=True,
                    )
                    st.caption(p["delivery_text"])

            st.write("")
            run_prediction_ui(
                amount=float(selected_product["price_inr"]),
                locked_amount=True,
                widget_prefix=f"product_{selected_product['id']}",
            )

        # Dừng ở đây để không render danh sách sản phẩm bên dưới
        st.stop()

    # Thanh breadcrumb mô phỏng giống Amazon (chỉ hiện ở subtab duyệt)
    st.markdown(
        """
        <div style="font-size: 0.9rem; color: #777; margin-bottom: 0.5rem;">
            Amazon.in &gt; Thời trang &gt; Quần áo &gt; <b>""" + selected_clothing_type + """</b>
        </div>
        """,
        unsafe_allow_html=True,
    )

    products_demo = [
        {
            "id": "tee_white_499",
            "name": "Áo thun nam basic cổ tròn",
            "subtitle": "Trắng / Cotton 100%",
            "price_inr": 499,
            "delivery_text": "Dự kiến giao: 3–5 ngày làm việc.",
            "image_url": "https://images.unsplash.com/photo-1521572163474-6864f9cf17ab?auto=format&fit=crop&w=400&q=80",
        },
        {
            "id": "hoodie_black_1299",
            "name": "Hoodie oversize unisex nỉ dày",
            "subtitle": "Đen / Form rộng",
            "price_inr": 1299,
            "delivery_text": "Dự kiến giao: 4–6 ngày làm việc.",
            "image_url": "https://images.unsplash.com/photo-1529927066849-66e1abc70a2e?auto=format&fit=crop&w=400&q=80",
        },
        {
            "id": "shirt_navy_899",
            "name": "Áo sơ mi tay dài slim fit",
            "subtitle": "Xanh navy",
            "price_inr": 899,
            "delivery_text": "Dự kiến giao: 2–4 ngày làm việc.",
            "image_url": "https://images.unsplash.com/photo-1528701800489-20be3c30c1d5?auto=format&fit=crop&w=400&q=80",
        },
        {
            "id": "jeans_2500",
            "name": "Quần jean nam dáng slim",
            "subtitle": "Xanh đậm / Co giãn nhẹ",
            "price_inr": 2500,
            "delivery_text": "Dự kiến giao: 3–5 ngày làm việc.",
            "image_url": "https://images.unsplash.com/photo-1542272604-787c3835535d?auto=format&fit=crop&w=400&q=80",
        },
        {
            "id": "tee_pink_749",
            "name": "Áo thun nữ oversize graphic",
            "subtitle": "Hồng pastel",
            "price_inr": 749,
            "delivery_text": "Dự kiến giao: 5–7 ngày làm việc.",
            "image_url": "https://images.unsplash.com/photo-1521572267360-ee0c2909d518?auto=format&fit=crop&w=400&q=80",
        },
    ]

    cols = st.columns(3)
    for idx, p in enumerate(products_demo):
        with cols[idx % 3]:
            with st.container(border=True):
                st.image(p["image_url"], use_container_width=True)
                st.markdown(f"**{p['name']}**")
                st.caption(p["subtitle"])
                st.markdown(
                    f"<span style='color:#B12704; font-weight:700;'>₹ {p['price_inr']:,} INR</span>",
                    unsafe_allow_html=True,
                )
                st.caption(p["delivery_text"])

                if st.button("Xem dự đoán", key=f"predict_{p['id']}", use_container_width=True):
                    st.session_state["selected_product"] = p
                    st.session_state["nav_to_page"] = "🛍️ Browser Sản Phẩm"
                    st.session_state["nav_to_subtab"] = "🎯 Dự đoán sản phẩm"
                    st.rerun()

    st.caption("Bấm “Xem dự đoán” để tự chuyển sang tab 🎯 Dự đoán sản phẩm.")
