import streamlit as st
import seaborn as sns
import pandas as pd
import numpy as np
import joblib
import lightgbm as lgb
import os
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime
from sqlalchemy import create_engine
import google.generativeai as genai
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import mutual_info_classif
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


def get_model_feature_names(m) -> list[str]:
    # Try sklearn API first
    names = getattr(m, "feature_names_in_", None)
    if names is not None:
        return [str(x) for x in list(names)]

    # LightGBM sklearn wrapper
    names = getattr(m, "feature_name_", None)
    if names is not None:
        return [str(x) for x in list(names)]

    booster = getattr(m, "booster_", None) or getattr(m, "_Booster", None)
    if booster is not None:
        try:
            return [str(x) for x in booster.feature_name()]
        except Exception:
            pass

    return []


@st.cache_data(show_spinner=False)
def load_category_options() -> list[str]:
    # Prefer CSV (local) for quick startup; fallback to a small default list
    try:
        if DATA_CSV_PATH.exists():
            df = pd.read_csv(DATA_CSV_PATH)
            for col in ["category", "Category"]:
                if col in df.columns:
                    opts = (
                        df[col]
                        .dropna()
                        .astype(str)
                        .map(lambda x: x.strip())
                        .loc[lambda s: s != ""]
                        .unique()
                        .tolist()
                    )
                    opts = sorted(opts)[:300]
                    if opts:
                        return opts
    except Exception:
        pass

    return ["Blouse", "Bottom", "Dress", "Dupatta", "Ethnic Dress", "Kurta", "Saree", "Set", "Top"]

# ==========================================
# DỮ LIỆU EDA (CSV) - Tự refresh theo mtime
# ==========================================
DATA_CSV_PATH = Path("data") / "ecommerce_orders.csv"

DATABASE_URI = os.environ.get(
    "DATABASE_URI",
    "postgresql://neondb_owner:npg_Oj2irQBMP0Xw@ep-restless-mouse-a1l98lhe-pooler.ap-southeast-1.aws.neon.tech/neondb?sslmode=require&channel_binding=require",
)


@st.cache_resource(show_spinner=False)
def get_db_engine(uri: str):
    try:
        return create_engine(uri, pool_pre_ping=True)
    except ModuleNotFoundError as e:
        # Thường gặp: thiếu psycopg2/psycopg driver
        raise e


@st.cache_data(show_spinner=False)
def load_orders_db(uri: str, refresh_token: int) -> pd.DataFrame:
    engine = get_db_engine(uri)
    df = pd.read_sql_table("ecommerce_orders", con=engine)
    if "Unnamed: 22" in df.columns:
        df = df.drop(columns=["Unnamed: 22"])
    return df


@st.cache_data(show_spinner=False)
def load_orders_csv(csv_path: str, mtime: float) -> pd.DataFrame:
    # mtime được truyền vào để cache tự invalidate khi file thay đổi
    df = pd.read_csv(csv_path)
    if "Unnamed: 22" in df.columns:
        df = df.drop(columns=["Unnamed: 22"])
    return df


def _to_bool(v) -> bool:
    if isinstance(v, bool):
        return v
    if v is None:
        return False
    s = str(v).strip().lower()
    return s in {"1", "true", "yes", "y", "t"}


def _size_to_ordinal(size_val) -> float:
    if size_val is None:
        return np.nan
    s = str(size_val).strip().upper().replace(" ", "")
    mapping = {
        "FREE": 0,
        "XS": 1,
        "S": 2,
        "M": 3,
        "L": 4,
        "XL": 5,
        "2XL": 6,
        "XXL": 6,
        "3XL": 7,
        "4XL": 8,
        "5XL": 9,
        "6XL": 10,
    }
    return mapping.get(s, np.nan)


def _safe_text(v) -> str:
    if v is None:
        return ""
    try:
        if pd.isna(v):
            return ""
    except Exception:
        pass
    return str(v)


def apply_auto_features(row: dict, df_columns: list[str]) -> dict:
    """Tự đồng bộ các cột feature khi lưu."""
    status = str(row.get("Status", "")).strip().lower()
    if "Status_binary" in df_columns:
        bad_markers = ("cancelled", "rejected", "returned", "return")
        row["Status_binary"] = 0 if any(m in status for m in bad_markers) else 1

    if "size_ordinal" in df_columns:
        row["size_ordinal"] = _size_to_ordinal(row.get("Size"))

    if "B2B_binary" in df_columns:
        row["B2B_binary"] = 1 if _to_bool(row.get("B2B")) else 0

    # ship-service-level tự cập nhật theo fulfilled-by (Easy Ship -> Standard, None -> Expedited)
    if "ship-service-level" in df_columns and "fulfilled-by" in df_columns:
        fb = str(row.get("fulfilled-by", "")).strip().lower()
        row["ship-service-level"] = "Standard" if fb == "easy ship" else "Expedited"

    if "fulfillment_binary" in df_columns:
        # Ưu tiên dùng fulfilled-by: Easy Ship = thường (0), rỗng/None = premium (1)
        if "fulfilled-by" in df_columns:
            fb = str(row.get("fulfilled-by", "")).strip().lower()
            row["fulfillment_binary"] = 0 if fb == "easy ship" else 1
        else:
            lvl = str(row.get("ship-service-level", "")).strip().lower()
            row["fulfillment_binary"] = 0 if lvl in {"", "standard"} else 1

    # ship_premium không chỉnh tay, đồng bộ theo fulfillment_binary (2 cái là 1)
    if "ship_premium" in df_columns and "fulfillment_binary" in df_columns:
        row["ship_premium"] = int(row.get("fulfillment_binary") or 0)

    return row


def get_gemini_api_key() -> str | None:
    # Prioritize env var then Streamlit secrets
    key = os.getenv("GEMINI_API_KEY")
    if key:
        return key.strip()
    try:
        key = st.secrets.get("gemini_api_key")
    except Exception:
        key = None
    if key:
        return str(key).strip()
    return None


def ask_gemini_chat(user_prompt: str, conversation: list[dict[str, str]] | None = None, model: str = "gemini-1.5-turbo") -> str:
    import json
    import requests

    api_key = get_gemini_api_key()
    if not api_key:
        raise ValueError("Gemini API key is not configured. Set GEMINI_API_KEY environment variable or st.secrets['gemini_api_key'].")

    if conversation is None:
        conversation = [
            {"role": "system", "content": "You are a helpful project assistant for an Amazon sales prediction dashboard. Answer concisely in Vietnamese when user asks in Vietnamese."}
        ]
    else:
        # include system intro in thread
        if not any(m["role"] == "system" for m in conversation):
            conversation = [
                {"role": "system", "content": "You are a helpful project assistant for an Amazon sales prediction dashboard. Answer concisely in Vietnamese when user asks in Vietnamese."}
            ] + conversation

    messages_payload = []
    for m in conversation:
        if m["role"] in {"system", "user", "assistant"}:
            if isinstance(m["content"], str):
                messages_payload.append({"role": m["role"], "content": {"text": m["content"]}})

    body = {
        "messages": messages_payload,
        "temperature": 0.2,
        "maxOutputTokens": 512,
    }

    url = f"https://gemini.googleapis.com/v1/models/{model}:generateMessage"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    resp = requests.post(url, headers=headers, json=body, timeout=30)
    if resp.status_code != 200:
        raise RuntimeError(f"Gemini API failed ({resp.status_code}): {resp.text}")
    data = resp.json()
    # Gemini returns candidates list
    candidates = data.get("candidates") or []
    if not candidates:
        # fallback for responses in message
        message_obj = data.get("message") or {}
        if isinstance(message_obj, dict):
            content = message_obj.get("content") or {}
            if isinstance(content, dict):
                return content.get("text", "[Không có phản hồi]")
        raise RuntimeError("Gemini API response has no candidates")
    first = candidates[0]
    if isinstance(first.get("content"), dict):
        return first["content"].get("text", "[Không có phản hồi]")
    return first.get("content", "[Không có phản hồi]")


SIZE_ORDER = ["Free", "XS", "S", "M", "L", "XL", "2XL", "3XL", "4XL", "5XL", "6XL"]

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


def run_prediction_ui(*, amount: float, locked_amount: bool, widget_prefix: str, locked_category: bool = False, category_value: str | None = None):
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
                category_opts = load_category_options()
                default_cat = "Blouse" if "Blouse" in category_opts else (category_opts[0] if category_opts else "Blouse")
                fixed_cat = str(category_value) if category_value is not None and str(category_value).strip() else None
                if fixed_cat:
                    if fixed_cat not in category_opts:
                        category_opts = [fixed_cat] + list(category_opts)
                    selected_cat = fixed_cat
                else:
                    selected_cat = default_cat
                category = st.selectbox(
                    "🏷️ Category",
                    options=category_opts if category_opts else [selected_cat],
                    index=(category_opts.index(selected_cat) if category_opts and selected_cat in category_opts else 0),
                    key=f"{widget_prefix}_category",
                    disabled=locked_category,
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

        if st.button("🚀 Chạy Mô Hình Dự Đoán", type="primary", use_container_width=True, key=f"{widget_prefix}_run"):
            feature_names = get_model_feature_names(model)
            expected_lower = [c.lower() for c in feature_names]

            def put(lower_name: str, fallback_name: str, value):
                if feature_names:
                    if lower_name in expected_lower:
                        actual = feature_names[expected_lower.index(lower_name)]
                        row[actual] = value
                else:
                    row[fallback_name] = value

            # Build row dynamically to avoid feature mismatch between model versions
            row: dict[str, object] = {}
            put("qty", "Qty", int(quantity))
            put("amount", "Amount", float(amount))
            put("fulfillment_binary", "fulfillment_binary", int(fulfillment_binary))
            put("b2b_binary", "B2B_binary", int(B2B_binary))
            put("size_ordinal", "size_ordinal", int(size_ordinal))
            put("category", "category", str(category))

            input_data = pd.DataFrame([row])

            # Ensure categorical dtype for LightGBM if needed
            for c in list(input_data.columns):
                if str(c).lower() == "category":
                    try:
                        input_data[c] = input_data[c].astype("category")
                    except Exception:
                        pass

            # If we can detect exact feature ordering, enforce it and fill missing with NaN
            if feature_names:
                for c in feature_names:
                    if c not in input_data.columns:
                        input_data[c] = np.nan
                input_data = input_data[feature_names]

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
    run_prediction_ui(amount=100.0, locked_amount=False, widget_prefix="playground", locked_category=False)

# ----------------------------------------------------------------------
# TAB 2: EXPLORATORY DATA ANALYSIS (EDA)
# ----------------------------------------------------------------------
elif active_page == "📊 Data Exploration":
    st.markdown("<h2 style='text-align: center; color: #4CAF50;'>📊 Khám Phá Dữ Liệu (EDA)</h2>", unsafe_allow_html=True)
    st.markdown(
        "<p style='text-align: center; color: #666; font-size: 1.05rem;'>"
        "Đọc dữ liệu và phân tích dữ liệu."
        "</p>",
        unsafe_allow_html=True,
    )
    st.divider()

    if "orders_refresh_token" not in st.session_state:
        st.session_state.orders_refresh_token = 0

    # Các nút thao tác EDA (khởi tạo trước, render sau khi có df)
    export_clicked = False
    download_csv_clicked = False

    if "orders_df" not in st.session_state:
        try:
            st.session_state.orders_df = load_orders_db(DATABASE_URI, int(st.session_state.orders_refresh_token))
        except ModuleNotFoundError as e:
            st.error(
                "Thiếu driver PostgreSQL để kết nối DB. Hãy cài dependencies rồi chạy lại:\n\n"
                "`pip install -r requirements.txt`\n\n"
                f"Chi tiết lỗi: `{e}`"
            )
            st.stop()
    df = st.session_state.orders_df

    # Info bar nhỏ gọn (đưa lên đầu + canh giữa)
    with st.container(border=True):
        st.markdown(
            f"""
            <div style="text-align:center; color:#ddd; font-size: 0.95rem;">
                <span><b>Số dòng:</b> {len(df):,}</span>
                &nbsp;&nbsp;|&nbsp;&nbsp;
                <span><b>Số cột:</b> {df.shape[1]:,}</span>
                &nbsp;&nbsp;|&nbsp;&nbsp;
                <span><b>Nguồn:</b> Server (PostgreSQL)</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # -------------------------
    # Thanh điều khiển EDA
    # -------------------------
    eda_nav_col, eda_content_col = st.columns([0.8, 2.2])
    with eda_nav_col:
        with st.container(border=True):
            st.markdown("#### 📊 EDA")
            eda_section = st.radio("Chọn mục", ["📄 Dữ liệu", "📈 Biểu đồ"], index=0, label_visibility="collapsed")
            st.markdown("---")
            export_clicked = st.button("⬆️ Tải lên DB", use_container_width=True, key="eda_export")
            
            # === NÚT TẢI CSV VỀ MÁY NGƯỜI DÙNG (thay vì ghi file trên server) ===
            st.download_button(
                label="⬇️ Tải dữ liệu CSV về Máy Tính",
                data=df.to_csv(index=False).encode('utf-8'),
                file_name=f"ecommerce_orders_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True,
                key="eda_download_csv"
            )

    # Export to server (replace table)
    if export_clicked:
        try:
            engine = get_db_engine(DATABASE_URI)
            df.to_sql("ecommerce_orders", con=engine, if_exists="replace", index=False)
            st.success("Đã export dữ liệu lên server (replace table `ecommerce_orders`).")
        except Exception as e:
            st.error(f"Export thất bại: {e}")

    # ======================
    # 1) DỮ LIỆU (CRUD UI)
    # ======================
    with eda_content_col:
        if eda_section == "📄 Dữ liệu":
            st.subheader("📄 Dữ liệu đơn hàng")
            st.caption("Xem danh sách đơn hàng. Nhấn **Chỉnh sửa** để mở menu cập nhật/xóa.")

            if "Order ID" not in df.columns:
                st.error("Không tìm thấy cột `Order ID` trong CSV để chỉnh sửa theo đơn hàng.")
                st.stop()

            if "orders_ui_mode" not in st.session_state:
                st.session_state.orders_ui_mode = "list"  # list | edit

            # ---------- LIST VIEW ----------
            if st.session_state.orders_ui_mode == "list":
                toolbar = st.columns([1.6, 1, 1.2])
                with toolbar[0]:
                    q = st.text_input("Tìm nhanh", placeholder="Order ID / Category / Status...")
                with toolbar[1]:
                    page_size = st.selectbox("Dòng / trang", [20, 50, 100], index=1)
                with toolbar[2]:
                    if st.button("➕ Thêm đơn", type="primary", use_container_width=True):
                        new_row = {c: "" for c in df.columns}
                        new_row["Order ID"] = f"NEW-{int(pd.Timestamp.now().timestamp())}"
                        # Defaults theo rule business
                        if "Status" in df.columns:
                            new_row["Status"] = "Cancelled"
                        if "Category" in df.columns:
                            new_row["Category"] = "Blouse"
                        if "Qty" in df.columns:
                            new_row["Qty"] = 1
                        if "currency" in df.columns:
                            new_row["currency"] = "INR"
                        if "Size" in df.columns:
                            new_row["Size"] = "Free"
                        if "fulfilled-by" in df.columns:
                            new_row["fulfilled-by"] = ""  # None = Premium
                        if "ship-service-level" in df.columns:
                            new_row["ship-service-level"] = "Expedited"
                        if "Sales Channel " in df.columns:
                            new_row["Sales Channel "] = "Amazon.in"
                        if "Fulfilment" in df.columns:
                            new_row["Fulfilment"] = "Merchant"
                        if "B2B" in df.columns:
                            new_row["B2B"] = False
                        if "index" in df.columns:
                            try:
                                new_row["index"] = int(pd.to_numeric(df["index"], errors="coerce").max() or -1) + 1
                            except Exception:
                                new_row["index"] = ""
                        st.session_state.orders_df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
                        st.session_state.selected_order_id = new_row["Order ID"]
                        st.session_state.orders_ui_mode = "edit"
                        st.rerun()

                # Pagination state (dùng input key riêng + nav flag để mũi tên hoạt động)
                if "orders_page_typed" not in st.session_state:
                    st.session_state.orders_page_typed = 1
                if "orders_nav_to_page" in st.session_state:
                    st.session_state.orders_page_typed = int(st.session_state.pop("orders_nav_to_page"))

                st.markdown("**Trang**")
                # Không nest columns quá 1 cấp: dùng 1 hàng columns duy nhất
                p1, p2, p3, p4, p5 = st.columns([0.6, 1.1, 0.8, 0.6, 2.2])
                with p1:
                    prev_clicked = st.button("◀", use_container_width=True, key="orders_page_prev")
                with p2:
                    typed_page = st.number_input(
                        "Nhập trang",
                        min_value=1,
                        value=int(st.session_state.orders_page_typed),
                        step=1,
                        label_visibility="collapsed",
                        key="orders_page_typed",
                    )
                total_label_placeholder = p3.empty()
                with p4:
                    next_clicked = st.button("▶", use_container_width=True, key="orders_page_next")
                with p5:
                    st.markdown(
                        "<div style='text-align:right; color:#777; padding-top: 28px;'>"
                        f"File: <code>{DATA_CSV_PATH.as_posix()}</code>"
                        "</div>",
                        unsafe_allow_html=True,
                    )

                work_df = df
                if q:
                    s = q.strip().lower()
                    mask = pd.Series(False, index=work_df.index)
                    for c in ["Order ID", "Category", "Status"]:
                        if c in work_df.columns:
                            mask = mask | work_df[c].astype(str).str.lower().str.contains(s, na=False)
                    work_df = work_df.loc[mask]

                # Sort theo cột "index" nếu có (đúng thứ tự dữ liệu gốc)
                if "index" in work_df.columns:
                    work_df = work_df.sort_values("index", ascending=True, kind="mergesort")

                total_rows = int(len(work_df))
                total_pages = max(1, int(np.ceil(total_rows / int(page_size))))

                # current page from typed input (clamp)
                current_page = int(typed_page)
                if current_page < 1:
                    current_page = 1
                if current_page > total_pages:
                    current_page = total_pages

                # Apply prev/next via nav flag (tránh set widget state sau khi render)
                if prev_clicked:
                    st.session_state["orders_nav_to_page"] = max(1, current_page - 1)
                    st.rerun()
                if next_clicked:
                    st.session_state["orders_nav_to_page"] = min(total_pages, current_page + 1)
                    st.rerun()

                # Update label
                total_label_placeholder.markdown(
                    f"<div style='text-align:center; padding-top:6px;'>/ {total_pages}</div>",
                    unsafe_allow_html=True,
                )

                start = (current_page - 1) * int(page_size)
                end = start + int(page_size)
                page_df = work_df.iloc[start:end].copy()

                # Hiển thị nhiều cột hơn (đã loại bỏ Unnamed: 22 khi load)
                display_cols = list(page_df.columns[:20])

                with st.container(border=True):
                    st.markdown("#### 📋 Danh sách đơn hàng")
                    # Click chọn row để auto chuyển sang chỉnh sửa (nếu Streamlit hỗ trợ selection)
                    try:
                        event = st.dataframe(
                            page_df[display_cols],
                            use_container_width=True,
                            hide_index=True,
                            on_select="rerun",
                            selection_mode="single-row",
                        )
                        if getattr(event, "selection", None) and event.selection.rows:
                            selected_row = page_df.iloc[int(event.selection.rows[0])]
                            st.session_state.selected_order_id = str(selected_row["Order ID"])
                            st.session_state.orders_ui_mode = "edit"
                            st.rerun()
                    except TypeError:
                        # Fallback cho bản Streamlit cũ
                        st.dataframe(page_df[display_cols], use_container_width=True, hide_index=True)

                if page_df.empty:
                    st.warning("Không có dữ liệu ở trang hiện tại.")
                    st.stop()

                # Fallback chọn đơn (khi không có click row)
                if "selected_order_id" not in st.session_state:
                    st.session_state.selected_order_id = str(page_df["Order ID"].astype(str).iloc[0])
                st.session_state.selected_order_id = st.selectbox(
                    "Chọn đơn để thao tác",
                    options=page_df["Order ID"].astype(str).tolist(),
                    index=page_df["Order ID"].astype(str).tolist().index(str(st.session_state.selected_order_id))
                    if str(st.session_state.selected_order_id) in page_df["Order ID"].astype(str).tolist()
                    else 0,
                )

                action1, action2, _ = st.columns([1, 1, 2])
                with action1:
                    # Click row sẽ tự vào chỉnh sửa (nếu Streamlit hỗ trợ selection).
                    # Fallback: dropdown "Chọn đơn để thao tác" ở trên sẽ đổi selected_order_id,
                    # và nút dưới đây cho phép vào edit nếu không click được.
                    if st.button("✏️ Chỉnh sửa", use_container_width=True, key="orders_list_edit_btn"):
                        st.session_state.orders_ui_mode = "edit"
                        st.rerun()
                with action2:
                    if st.button("🔄 Refresh", use_container_width=True, key="orders_list_refresh_db"):
                        st.session_state.orders_refresh_token = int(st.session_state.get("orders_refresh_token", 0)) + 1
                        st.session_state.pop("orders_df", None)
                        st.rerun()

            # ---------- EDIT VIEW ----------
            else:
                selected_order_id = str(st.session_state.get("selected_order_id", ""))
                if not selected_order_id:
                    st.session_state.orders_ui_mode = "list"
                    st.rerun()

                # header nav
                nav1, nav2, nav3 = st.columns([1, 1, 2])
                with nav1:
                    if st.button("← Back", use_container_width=True):
                        st.session_state.orders_ui_mode = "list"
                        st.rerun()
                with nav2:
                    if st.button("Cancel", use_container_width=True):
                        # không ghi file, chỉ quay lại list
                        st.session_state.orders_ui_mode = "list"
                        st.rerun()
                with nav3:
                    st.markdown(
                        "<div style='text-align:right; color:#777; padding-top: 6px;'>"
                        f"Đang chỉnh: <code>{selected_order_id}</code>"
                        "</div>",
                        unsafe_allow_html=True,
                    )

                # Lấy dòng đầu tiên match Order ID (nếu trùng ID sẽ edit theo dòng đầu tiên)
                matches = df.index[df["Order ID"].astype(str) == selected_order_id].tolist()
                if not matches:
                    st.error("Không tìm thấy Order ID trong dữ liệu hiện tại.")
                    st.session_state.orders_ui_mode = "list"
                    st.rerun()
                row_idx = int(matches[0])
                row = df.loc[row_idx].to_dict()

                core_cols = [
                    c
                    for c in [
                        "Order ID",
                        "Date",
                        "Status",
                        "Category",
                        "Size",
                        "Qty",
                        "Amount",
                        "currency",
                        "Sales Channel ",
                        "Fulfilment",
                        "fulfilled-by",
                        "ship-service-level",
                        "B2B",
                    ]
                    if c in df.columns
                ]
                # Model mới bỏ hẳn promotion => không cho chỉnh/hiện promotion nữa
                other_cols = [c for c in df.columns if c not in core_cols and c != "promotion"]
                locked_feature_cols = [
                    c
                    for c in ["Status_binary", "size_ordinal", "B2B_binary", "fulfillment_binary", "ship_premium"]
                    if c in df.columns
                ]
                editable_other_cols = [c for c in other_cols if c not in locked_feature_cols]

                st.markdown("#### 🧾 Menu chỉnh sửa")

                # Seed state cho editor để thay đổi là rerun (auto update vector)
                editor_key_prefix = f"order_edit::{selected_order_id}::"
                if st.session_state.get(editor_key_prefix + "__seeded") != True:
                    for col in core_cols + editable_other_cols:
                        raw = row.get(col, "")
                        if col == "Qty":
                            try:
                                st.session_state[editor_key_prefix + col] = int(float(raw))
                            except Exception:
                                st.session_state[editor_key_prefix + col] = 1
                            if int(st.session_state[editor_key_prefix + col]) < 1:
                                st.session_state[editor_key_prefix + col] = 1
                        elif col == "Amount":
                            try:
                                st.session_state[editor_key_prefix + col] = float(raw)
                            except Exception:
                                st.session_state[editor_key_prefix + col] = 0.0
                        elif col == "currency":
                            st.session_state[editor_key_prefix + col] = "INR"
                        elif col == "Size":
                            sval = _safe_text(raw).strip()
                            st.session_state[editor_key_prefix + col] = sval if sval in SIZE_ORDER else "Free"
                        elif col == "fulfilled-by":
                            fb = _safe_text(raw).strip()
                            st.session_state[editor_key_prefix + col] = "Easy Ship" if fb == "Easy Ship" else ""
                        elif col == "ship-service-level":
                            lvl = _safe_text(raw).strip()
                            st.session_state[editor_key_prefix + col] = "Expedited" if lvl == "Expedited" else "Standard"
                        elif col == "Sales Channel ":
                            st.session_state[editor_key_prefix + col] = "Amazon.in"
                        elif col == "Fulfilment":
                            f = _safe_text(raw).strip()
                            st.session_state[editor_key_prefix + col] = f if f in {"Amazon", "Merchant"} else "Merchant"
                        elif col == "B2B":
                            st.session_state[editor_key_prefix + col] = _to_bool(raw)
                        elif col == "Status":
                            sval = _safe_text(raw).strip()
                            st.session_state[editor_key_prefix + col] = sval if sval else "Cancelled"
                        elif col == "Category":
                            cval = _safe_text(raw).strip()
                            st.session_state[editor_key_prefix + col] = cval if cval else "Blouse"
                        elif col in ["Size", "currency", "fulfilled-by"]:
                            st.session_state[editor_key_prefix + col] = _safe_text(raw)
                        else:
                            # text_input yêu cầu state là string
                            st.session_state[editor_key_prefix + col] = _safe_text(raw)
                    st.session_state[editor_key_prefix + "__seeded"] = True

                c1, c2 = st.columns(2)
                edited_row = row.copy()
                for i, col in enumerate(core_cols):
                    target = c1 if i % 2 == 0 else c2
                    with target:
                        key = editor_key_prefix + col
                        val = st.session_state.get(key, "")
                        if col == "Qty":
                            try:
                                ival = int(float(val))
                            except Exception:
                                ival = 1
                            if ival < 1:
                                ival = 1
                            edited_row[col] = st.number_input(col, min_value=1, value=ival, step=1, key=key)
                        elif col == "Amount":
                            try:
                                fval = float(val)
                            except Exception:
                                fval = 0.0
                            edited_row[col] = st.number_input(col, min_value=0.0, value=fval, step=10.0, format="%.2f", key=key)
                        elif col in ["Status", "Category", "Size", "currency", "fulfilled-by", "ship-service-level", "Sales Channel ", "Fulfilment", "B2B"]:
                            if col == "currency":
                                edited_row[col] = st.text_input(col, value="INR", disabled=True, key=key)
                                continue
                            if col == "Size":
                                sval = str(val).strip() if val is not None else ""
                                default_size = sval if sval in SIZE_ORDER else "Free"
                                edited_row[col] = st.selectbox(col, options=SIZE_ORDER, index=SIZE_ORDER.index(default_size), key=key)
                                continue
                            if col == "fulfilled-by":
                                # Easy Ship = thường, rỗng = premium
                                opts = ["Easy Ship", ""]
                                sval = str(val).strip() if val is not None else ""
                                edited_row[col] = st.selectbox(
                                    col,
                                    options=opts,
                                    format_func=lambda x: "Easy Ship (Thường)" if str(x).strip() == "Easy Ship" else "Premium (Không Easy Ship)",
                                    index=0 if sval == "Easy Ship" else 1,
                                    key=key,
                                )
                                continue
                            if col == "ship-service-level":
                                # auto update theo fulfilled-by, không cho chỉnh tay
                                fb_now = str(edited_row.get("fulfilled-by", "")).strip()
                                lvl = "Standard" if fb_now == "Easy Ship" else "Expedited"
                                edited_row[col] = st.text_input(col, value=lvl, disabled=True, key=key)
                                continue
                            if col == "Sales Channel ":
                                edited_row[col] = st.text_input(col, value="Amazon.in", disabled=True, key=key)
                                continue
                            if col == "Fulfilment":
                                opts = ["Amazon", "Merchant"]
                                sval = str(val).strip() if val is not None else ""
                                if sval not in opts:
                                    sval = "Merchant"
                                edited_row[col] = st.selectbox(col, options=opts, index=opts.index(sval), key=key)
                                continue
                            if col == "B2B":
                                b2b_val = _to_bool(val)
                                edited_row[col] = st.selectbox(
                                    "B2B",
                                    options=[False, True],
                                    format_func=lambda x: "False" if x is False else "True",
                                    index=1 if b2b_val else 0,
                                    key=key,
                                )
                                continue

                            opts = sorted(df[col].dropna().astype(str).unique().tolist())[:200] if col in df.columns else []
                            sval = str(val) if val is not None else ""
                            if col == "Status":
                                sval = sval.strip() or "Cancelled"
                            if col == "Category":
                                sval = sval.strip() or "Blouse"
                            if sval and sval not in opts:
                                opts = [sval] + opts
                            edited_row[col] = st.selectbox(col, options=opts if opts else [sval], index=0, key=key)
                        else:
                            edited_row[col] = st.text_input(col, value=str(val) if val is not None else "", key=key)

                with st.expander("Trường nâng cao (tuỳ chọn)"):
                    adv_cols = st.columns(2)
                    for i, col in enumerate(editable_other_cols):
                        target = adv_cols[i % 2]
                        with target:
                            key = editor_key_prefix + col
                            val = st.session_state.get(key, "")
                            edited_row[col] = st.text_input(
                                col,
                                value=str(val) if val is not None else "",
                                disabled=(col == "index"),
                                key=key,
                            )

                computed = apply_auto_features(edited_row.copy(), list(df.columns))
                vector_fields = [
                    ("Qty", computed.get("Qty", "—")),
                    ("Amount", computed.get("Amount", "—")),
                    ("fulfillment_binary", computed.get("fulfillment_binary", "—")),
                    ("Category", computed.get("Category", computed.get("category", "—"))),
                    ("size_ordinal", computed.get("size_ordinal", "—")),
                    ("B2B_binary", computed.get("B2B_binary", "—")),
                ]
                with st.container(border=True):
                    st.markdown("#### 📌 Thông số vector")
                    st.dataframe(
                        pd.DataFrame(vector_fields, columns=["Feature", "Value"]),
                        use_container_width=True,
                        hide_index=True,
                    )
                    st.caption(f"**Status_binary:** {computed.get('Status_binary', '—')}")

                save_btn, delete_btn = st.columns([1, 1])
                do_save = save_btn.button("💾 Lưu", type="primary", use_container_width=True, key=editor_key_prefix + "__save")
                do_delete = delete_btn.button("🗑️ Xóa đơn", use_container_width=True, key=editor_key_prefix + "__delete")

                if do_delete:
                    st.session_state.orders_df = df.drop(index=row_idx).reset_index(drop=True)
                    st.success(f"Đã xóa đơn `{selected_order_id}` (chưa lưu vào file).")
                    st.session_state.orders_ui_mode = "list"
                    st.rerun()

                if do_save:
                    new_df = df.copy()
                    # Auto update các feature cốt lõi (không cho sửa tay)
                    computed = apply_auto_features(computed, list(df.columns))
                    for col in df.columns:
                        new_df.at[row_idx, col] = computed.get(col, new_df.at[row_idx, col])
                    new_df.to_csv(DATA_CSV_PATH, index=False)
                    load_orders_csv.clear()
                    st.session_state.orders_df = load_orders_csv(str(DATA_CSV_PATH), os.path.getmtime(DATA_CSV_PATH))
                    st.success("Đã lưu thay đổi vào CSV.")
                    st.session_state.orders_ui_mode = "list"
                    st.rerun()

    # ======================
    # 2) BIỂU ĐỒ (BOXPLOT)
    # ======================
        else:
            st.subheader("📈 Khám phá dữ liệu nâng cao (EDA Analytics)")
            st.caption("Khu vực này thực hiện các tính toán nặng (Pearson Correlation, Mutual Information). Bấm nút bên dưới để bắt đầu phân tích.")
            
            # Đảm bảo có Status_binary để tính toán
            work_df = df.copy()
            if "Status_binary" not in work_df.columns and "Status" in work_df.columns:
                bad_markers = ("cancelled", "rejected", "returned", "return")
                work_df["Status_binary"] = work_df["Status"].astype(str).str.lower().apply(
                    lambda x: 0 if any(m in x for m in bad_markers) else 1
                )

            # Nút Trigger
            run_charts = st.button("🚀 RUN CHART", type="primary", use_container_width=True)

            if run_charts:
                with st.spinner("Đang xử lý dữ liệu và vẽ biểu đồ. Vui lòng đợi (có thể mất vài giây)..."):
                    
                    # ---------------------------------------------------------
                    # TEXT METRICS (Thay cho các lệnh print)
                    # ---------------------------------------------------------
                    st.markdown("### 📌 Thông tin tổng quan")
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Số States (Tỉnh/Bang)", work_df['ship-state'].nunique() if 'ship-state' in work_df.columns else 0)
                    c2.metric("Số Cities (Thành phố)", work_df['ship-city'].nunique() if 'ship-city' in work_df.columns else 0)
                    c3.metric("Số đơn B2B", int(work_df['B2B_binary'].sum()) if 'B2B_binary' in work_df.columns else 0)

                    s_col1, s_col2 = st.columns([1, 1])
                    with s_col1:
                        st.markdown("**Tỷ lệ thành công theo Size:**")
                        if 'size_ordinal' in work_df.columns:
                            size_success = work_df.groupby('size_ordinal')['Status_binary'].mean().reset_index()
                            size_success.columns = ['Size Ordinal', 'Tỷ lệ thành công']
                            st.dataframe(size_success, hide_index=True, use_container_width=True)
                    
                    with s_col2:
                        st.markdown("**Số lượng đơn theo các size đặc biệt:**")
                        if 'size_ordinal' in work_df.columns:
                            s8 = work_df[work_df['size_ordinal'] == 8].shape[0]
                            s9 = work_df[work_df['size_ordinal'] == 9].shape[0]
                            s10 = work_df[work_df['size_ordinal'] == 10].shape[0]
                            st.info(f"Size 8 (4XL): **{s8}** đơn\n\nSize 9 (5XL): **{s9}** đơn\n\nSize 10 (6XL): **{s10}** đơn")

                    st.divider()

                    # ---------------------------------------------------------
                    # PLOT 1: 2x2 Grid Tổng quan
                    # ---------------------------------------------------------
                    st.markdown("### 📊 Tổng quan Kinh doanh (2x2 Grid)")
                    
                    # Xử lý Date an toàn hơn một chút
                    if 'Date' in work_df.columns:
                        work_df['Date'] = pd.to_datetime(work_df['Date'], errors='coerce')

                    fig1, axes = plt.subplots(2, 2, figsize=(16, 12))
                    sns.set_theme(style="whitegrid")

                    # 1. Average Price: B2B vs Non-B2B
                    if 'B2B_binary' in work_df.columns and 'Amount' in work_df.columns:
                        avg_price_b2b = work_df.groupby('B2B_binary')['Amount'].mean().reset_index()
                        avg_price_b2b['B2B_Type'] = avg_price_b2b['B2B_binary'].map({0: 'Regular Customer', 1: 'B2B (Business)'})
                        sns.barplot(x='B2B_Type', y='Amount', data=avg_price_b2b, ax=axes[0, 0], palette='Blues_d')
                        axes[0, 0].set_title('1. Average Order Amount (B2B vs Regular)', fontsize=14, fontweight='bold')
                        axes[0, 0].set_ylabel('Average Amount (INR)', fontsize=12)
                        axes[0, 0].set_xlabel('')
                        for i, v in enumerate(avg_price_b2b['Amount']):
                            axes[0, 0].text(i, v + 5, f'₹{v:.2f}', ha='center', fontsize=12, fontweight='bold')

                    # 2. Average Success Rate by Category
                    if 'Category' in work_df.columns:
                        overall_success = work_df['Status_binary'].mean() * 100
                        cat_success = work_df.groupby('Category')['Status_binary'].mean().reset_index()
                        cat_success['Success Rate (%)'] = cat_success['Status_binary'] * 100
                        cat_success = cat_success.sort_values(by='Success Rate (%)', ascending=False)
                        sns.barplot(x='Success Rate (%)', y='Category', data=cat_success, ax=axes[0, 1], palette='Greens_r')
                        axes[0, 1].set_title(f'2. Success Rate by Top Categories\n(Overall Average: {overall_success:.1f}%)', fontsize=14, fontweight='bold')
                        axes[0, 1].set_xlabel('Success Rate (%)', fontsize=12)
                        axes[0, 1].set_ylabel('')
                        axes[0, 1].set_xlim(0, 100)
                        for i, v in enumerate(cat_success['Success Rate (%)']):
                            axes[0, 1].text(v - 8, i, f'{v:.1f}%', va='center', color='white', fontweight='bold')
                    else:
                        axes[0, 1].set_title('2. Average Success Rate by Category (N/A)', fontsize=14, fontweight='bold')
                        axes[0, 1].text(0.5, 0.5, "Column 'Category' not found.", ha='center', va='center', fontsize=12, color='red')

                    # 3. Monthly Sales Revenue Trend
                    if 'Date' in work_df.columns and 'Amount' in work_df.columns:
                        work_df['Month_Year'] = work_df['Date'].dt.to_period('M').astype(str)
                        monthly_sales = work_df.dropna(subset=['Month_Year']).groupby('Month_Year')['Amount'].sum().reset_index()
                        monthly_sales = monthly_sales.sort_values('Month_Year')
                        sns.lineplot(x='Month_Year', y='Amount', data=monthly_sales, ax=axes[1, 0], marker='o', color='purple', linewidth=2.5, markersize=8)
                        axes[1, 0].set_title('3. Total Sales Revenue Over Time', fontsize=14, fontweight='bold')
                        axes[1, 0].set_xlabel('Month', fontsize=12)
                        axes[1, 0].set_ylabel('Total Revenue (INR)', fontsize=12)
                        axes[1, 0].tick_params(axis='x', rotation=45)
                        axes[1, 0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}M".format(x/1e6)))

                    # 4. Top 10 States by Order Count
                    if 'ship-state' in work_df.columns:
                        top_states = work_df['ship-state'].value_counts().nlargest(10).reset_index()
                        top_states.columns = ['State', 'Total Orders']
                        sns.barplot(x='Total Orders', y='State', data=top_states, ax=axes[1, 1], palette='rocket')
                        axes[1, 1].set_title('4. Top 10 States by Order Volume', fontsize=14, fontweight='bold')
                        axes[1, 1].set_xlabel('Total Orders', fontsize=12)
                        axes[1, 1].set_ylabel('')

                    plt.tight_layout()
                    st.pyplot(fig1)
                    plt.close(fig1) # Đóng fig để tránh rò rỉ bộ nhớ của Streamlit

                    st.divider()

                    # ---------------------------------------------------------
                    # PLOT 2: Mutual Information (Categorical)
                    # ---------------------------------------------------------
                    st.markdown("### 🧠 Feature Importance: Mutual Information (Categorical)")
                    cat_cols = ['ship-state', 'Courier Status', 'Category', 'Sales Channel ', 'ship-city', 'Style', 'SKU']
                    existing_cat_cols = [col for col in cat_cols if col in work_df.columns]
                    
                    if existing_cat_cols:
                        sample_df = work_df[['Status_binary'] + existing_cat_cols].copy().dropna(subset=['Status_binary']).sample(n=min(20000, len(work_df)), random_state=42)
                        le = LabelEncoder()
                        for col in existing_cat_cols:
                            sample_df[col] = sample_df[col].fillna('Unknown')
                            sample_df[col] = le.fit_transform(sample_df[col].astype(str))

                        X = sample_df[existing_cat_cols]
                        y = sample_df['Status_binary']
                        
                        mi_scores_cat = mutual_info_classif(X, y, random_state=42)
                        mi_series_cat = pd.Series(mi_scores_cat, index=X.columns).sort_values(ascending=False)

                        fig2, ax2 = plt.subplots(figsize=(10, 6))
                        sns.barplot(x=mi_series_cat.values, y=mi_series_cat.index, palette='rocket', ax=ax2)
                        ax2.set_title('Mutual Information with Target (Categorical Columns)', fontsize=14)
                        ax2.set_xlabel('MI Score (Importance)', fontsize=12)
                        ax2.set_ylabel('Features', fontsize=12)
                        ax2.grid(axis='x', linestyle='--', alpha=0.7)
                        
                        plt.tight_layout()
                        st.pyplot(fig2)
                        plt.close(fig2)
                    else:
                        st.warning("Không tìm thấy các cột Categorical để tính toán MI.")

                    st.divider()

                    # ---------------------------------------------------------
                    # PLOT 3: Pearson vs Mutual Information (Numerical)
                    # ---------------------------------------------------------
                    st.markdown("### 🔍 Correlation vs Mutual Information (Numerical)")
                    features = ['Qty', 'Amount', 'fulfillment_binary', 'ship_premium', 'size_ordinal', 'B2B_binary']
                    existing_num_cols = [col for col in features if col in work_df.columns]

                    if existing_num_cols:
                        # Pearson
                        corr_matrix = work_df[['Status_binary'] + existing_num_cols].corr()
                        pearson_corr = corr_matrix['Status_binary'].drop('Status_binary').sort_values(ascending=False)

                        # MI
                        sample_df_num = work_df[['Status_binary'] + existing_num_cols].dropna().sample(n=min(20000, len(work_df)), random_state=42)
                        X_num = sample_df_num[existing_num_cols]
                        y_num = sample_df_num['Status_binary']
                        
                        mi_scores_num = mutual_info_classif(X_num, y_num, random_state=42)
                        mi_series_num = pd.Series(mi_scores_num, index=X_num.columns).sort_values(ascending=False)

                        fig3, axes3 = plt.subplots(1, 2, figsize=(15, 6))

                        # Plot 1: Pearson
                        sns.barplot(x=pearson_corr.values, y=pearson_corr.index, ax=axes3[0], palette='viridis')
                        axes3[0].set_title('Pearson Correlation with Target', fontsize=14)
                        axes3[0].set_xlabel('Correlation Coefficient', fontsize=12)
                        axes3[0].set_ylabel('Features', fontsize=12)
                        axes3[0].grid(axis='x', linestyle='--', alpha=0.7)

                        # Plot 2: MI
                        sns.barplot(x=mi_series_num.values, y=mi_series_num.index, ax=axes3[1], palette='magma')
                        axes3[1].set_title('Mutual Information Scores with Target', fontsize=14)
                        axes3[1].set_xlabel('MI Score', fontsize=12)
                        axes3[1].set_ylabel('')
                        axes3[1].grid(axis='x', linestyle='--', alpha=0.7)

                        plt.tight_layout()
                        st.pyplot(fig3)
                        plt.close(fig3)
                    else:
                        st.warning("Không tìm thấy các cột Numerical để tính toán.")
                        # ---------------------------------------------------------
                    # TEXT METRICS & BẢNG THỐNG KÊ (Mới thêm)
                    # ---------------------------------------------------------
                    st.markdown("### 📌 Thông tin tổng quan")
                    
                    # Tính toán tổng doanh thu và AOV (Average Order Value)
                    total_revenue = work_df['Amount'].sum() if 'Amount' in work_df.columns else 0
                    aov = work_df['Amount'].mean() if 'Amount' in work_df.columns else 0

                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Tổng Doanh Thu (INR)", f"₹ {total_revenue:,.0f}")
                    c2.metric("Giá Trị Đơn TB (AOV)", f"₹ {aov:,.0f}")
                    c3.metric("Số States (Tỉnh/Bang)", work_df['ship-state'].nunique() if 'ship-state' in work_df.columns else 0)
                    c4.metric("Số đơn B2B", int(work_df['B2B_binary'].sum()) if 'B2B_binary' in work_df.columns else 0)

                    st.divider()

                    # ---------------------------------------------------------
                    # BẢNG THỐNG KÊ DOANH THU (Tháng & Danh mục)
                    # ---------------------------------------------------------
                    st.markdown("### 💰 Thống kê chi tiết Doanh Thu")
                    rev_col1, rev_col2 = st.columns(2)

                    with rev_col1:
                        st.markdown("**1. Hiệu suất theo Tháng:**")
                        if 'Date' in work_df.columns and 'Amount' in work_df.columns:
                            work_df['Month_Year'] = pd.to_datetime(work_df['Date'], errors='coerce').dt.to_period('M').astype(str)
                            monthly_rev = work_df.groupby('Month_Year').agg(
                                Tổng_Doanh_Thu=('Amount', 'sum'),
                                Số_Đơn_Hàng=('Order ID', 'count')
                            ).reset_index()
                            # Format tiền tệ cho đẹp
                            monthly_rev_display = monthly_rev.copy()
                            monthly_rev_display['Tổng_Doanh_Thu'] = monthly_rev_display['Tổng_Doanh_Thu'].apply(lambda x: f"₹ {x:,.0f}")
                            st.dataframe(monthly_rev_display, hide_index=True, use_container_width=True)

                    with rev_col2:
                        st.markdown("**2. Doanh thu theo Danh mục (Category):**")
                        if 'Category' in work_df.columns and 'Amount' in work_df.columns:
                            cat_rev = work_df.groupby('Category').agg(
                                Tổng_Doanh_Thu=('Amount', 'sum'),
                                Tỷ_Lệ_Thành_Công=('Status_binary', lambda x: f"{(x.mean() * 100):.1f}%")
                            ).reset_index()
                            cat_rev = cat_rev.sort_values(by='Tổng_Doanh_Thu', ascending=False)
                            cat_rev['Tổng_Doanh_Thu'] = cat_rev['Tổng_Doanh_Thu'].apply(lambda x: f"₹ {x:,.0f}")
                            st.dataframe(cat_rev, hide_index=True, use_container_width=True)

                    st.divider()

                    # ---------------------------------------------------------
                    # INSIGHT VẬN HÀNH: AMAZON VS MERCHANT
                    # ---------------------------------------------------------
                    st.markdown("### 🚚 Phân tích rủi ro Vận Hành (Fulfillment Insights)")
                    if 'Fulfilment' in work_df.columns and 'Status_binary' in work_df.columns:
                        ff_col1, ff_col2 = st.columns([1, 1.5])
                        
                        ff_data = work_df.groupby('Fulfilment').agg(
                            Số_Đơn=('Order ID', 'count'),
                            Thành_Công=('Status_binary', 'sum')
                        ).reset_index()
                        ff_data['Tỷ_Lệ_Thành_Công_Vận_Chuyển'] = (ff_data['Thành_Công'] / ff_data['Số_Đơn']) * 100
                        
                        with ff_col1:
                            st.markdown("**Thống kê Fulfillment:**")
                            display_ff = ff_data[['Fulfilment', 'Số_Đơn', 'Tỷ_Lệ_Thành_Công_Vận_Chuyển']].copy()
                            display_ff['Tỷ_Lệ_Thành_Công_Vận_Chuyển'] = display_ff['Tỷ_Lệ_Thành_Công_Vận_Chuyển'].apply(lambda x: f"{x:.2f}%")
                            st.dataframe(display_ff, hide_index=True, use_container_width=True)
                            st.info("💡 **Business Insight:** Sự chênh lệch tỷ lệ thành công giữa Amazon và Merchant là rất lớn. Đây là rủi ro chính dẫn đến hủy đơn cần được can thiệp sớm.")

                        with ff_col2:
                            fig_ff, ax_ff = plt.subplots(figsize=(7, 3.5))
                            sns.barplot(x='Tỷ_Lệ_Thành_Công_Vận_Chuyển', y='Fulfilment', data=ff_data, palette='Set2', ax=ax_ff)
                            ax_ff.set_title('Tỷ lệ Giao Hàng Thành Công (Amazon vs Merchant)', fontsize=12, fontweight='bold')
                            ax_ff.set_xlabel('Tỷ lệ thành công (%)')
                            ax_ff.set_ylabel('')
                            ax_ff.set_xlim(0, 100)
                            for i, v in enumerate(ff_data['Tỷ_Lệ_Thành_Công_Vận_Chuyển']):
                                ax_ff.text(v - 8 if v > 15 else v + 2, i, f'{v:.1f}%', va='center', color='white' if v > 15 else 'black', fontweight='bold')
                            st.pyplot(fig_ff)
                            plt.close(fig_ff)
                    else:
                        st.caption("Không tìm thấy cột 'Fulfilment' để phân tích vận hành.")

                    st.divider()

                    # ---------------------------------------------------------
                    # PLOT 1: 2x2 Grid Tổng quan (Giữ nguyên code cũ của bạn từ đây)
                    # ---------------------------------------------------------
                    st.markdown("### 📊 Tổng quan Kinh doanh (2x2 Grid)")
                    
                    # Xử lý Date an toàn hơn một chút
                    if 'Date' in work_df.columns:
                        work_df['Date'] = pd.to_datetime(work_df['Date'], errors='coerce')

                    fig1, axes = plt.subplots(2, 2, figsize=(16, 12))
                    sns.set_theme(style="whitegrid")

# ----------------------------------------------------------------------
# TAB 3: PROJECT ASSISTANT (CHATBOT)
# ----------------------------------------------------------------------
elif active_page == "💬 Project Assistant":
    st.header("💬 Project Assistant")
    st.write("Ask me anything about the data preparation, model training, or insights from this project!")
    
    # 1. Securely configure the Gemini API key
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    
    # 2. Define the bot's persona and rules
    project_rules =  f"""
    You are an AI assistant for a data science project created by a group of 6 students at FPT University.
    Project lead: Phan Văn Tú - SE205062
    Modeling: Phạm Lê Minh - missing student ID, Phan Văn Tú, Nguyễn Hoàng Khang - SE192093
    EDA: Phan Văn Tú, Quách Hoàng Minh - SE205193, Nguyễn Nguyên Vũ - SE200003
    Deploy: Phan Văn Tú, Phạm Lê Minh, Bùi Đình Khoa - SE205193
    The project predicts e-commerce order cancellation probabilities using a LightGBM model. 
    Contact info: phanvantu712@gmail.com
    
    Here is the complete project report to use as your knowledge base. Use this to answer user questions:
    
    KNOWLEDGE BASE:
    DAP391m – REPORT CARD

I.	INTRODUCTION
UNDERSTANDING THE E-COMMERCE BUSINESS
1.	Direct Financial Loss (Reverse Logistics): 
When an order is processed, shipped, and then cancelled or fails to deliver, the platform absorbs the transportation and handling costs. This "reverse logistics" process eats directly into profit margins because the company pays for shipping twice (out and back) with zero revenue to show for it.
2.	Operational Inefficiency and Opportunity Cost: 
Cancellations waste valuable human labor. Employees spend time picking, packing, and eventually unpacking and restocking these goods. Furthermore, while that item is stuck in transit or processing for a cancelled order, it is tied up in inventory and cannot be sold to a genuine buyer.
3.	Decreased Marketing ROI: 
Advertising budgets are spent to acquire customers and drive conversions. When a successfully converted order is cancelled, the Customer Acquisition Cost (CAC) for that transaction is entirely wasted, lowering the overall return on investment for marketing campaigns.
WHAT DOES THIS PROJECT DO?
1.	Data-driven strategy
Before predicting future cancellations, this project utilizes extensive Exploratory Data Analysis (EDA) to identify the core drivers of current failures. By analyzing customer behavior, the data reveals that shipping speed and promotional offers are critical factors in finalizing a sale, providing immediate strategic value for marketing and logistics teams.
2.	Proactive Intervention
A Random Forest predictive model is deployed as an early-warning system. By calculating the cancellation probability of incoming orders, the model allows the platform to take proactive, automated steps. High-risk orders can trigger targeted interventions, such as automated confirmation emails, personalized reminders, or even dynamic promotional offers to secure the buyer's commitment before the cancellation occurs.
3.	Scalability and Feature Importance
Random Forest algorithm inherently calculates feature importance, continuously validating the trends observed in the initial EDA. As market dynamics shift—such as changes in buyer expectations for expedited shipping—the model is highly scalable. It can be continuously fine-tuned with incoming data, ensuring the platform's intervention strategies evolve alongside customer behavior.
II.              METHODOLOGIES
1.	Dataset Description & Technical Stack 

Sourced from Kaggle, representing transactions from an Indian clothing retailer in 2022. The raw dataset contained 128,975 records across 23 features. The technical workflow was supported by a robust data pipeline and analytical stack:
2.	Database Management
PostgreSQL was utilized to store and query the cleaned dataset efficiently.
3.	Development & Deployment
Python (with libraries like Pandas, Scikit-learn, Psycogs, matplotlib, joblib) was used for data manipulation, modeling, data exploration and providing charts. With GitHub for version control and Streamlit for deploying the final interactive application.
4.	Analytical Techniques: 
The research employed K-means clustering, Pairplots, Pearson/multivariable correlation, and robust model validation techniques like K-fold cross-validation.
5.	Note: AI assistants were utilized for code debugging and structural refinement. 
DATA CLEANING AND PREPROCESSING 
To prepare the raw data for the Random Forest algorithm, a rigorous preprocessing pipeline was implemented, reducing the dataset from 128, 976 rows to a clean, finalized count of 128,036 rows. The process prioritized the features most relevant to predicting order success, leaving extraneous columns in their raw state to preserve data integrity for separate exploratory analysis. Key steps included:
•	Target Variable Definition: The Status column was converted into a binary target variable (Status_binary), where successful deliveries (e.g., 'Shipped', 'Delivered to Buyer') were mapped to 1, and failed/cancelled orders were mapped to 0. Ambiguous 'Pending' orders were dropped from the dataset as their final classification was unknown.
•	Addressing Data Leakage: Anomalies in the Qty (Quantity) column where values equaled 0 were imputed to 1. This prevents data leakage and logical inconsistencies, as an order cannot exist with zero items. (And possibly Promotion_ID, we will address this further down the report.)
•	Targeted Imputation: Missing values in the financial Amount column (7,795 nulls) were imputed using the median amount grouped by the item's Size. A global median fallback was applied to capture any remaining nulls, ensuring the financial data remained robust without skewing the distribution.
•	Feature Engineering & Encoding: Categorical and text-based features were transformed into numerical formats for model compatibility:
o	Binary Encoding: Fulfilment (Amazon vs. Merchant), ship-service-level (Expedited vs. Standard), and B2B status were mapped to 0 and 1.
o	Ordinal Encoding: The Size column was mapped to an ordinal scale (0 to 10, representing 'Free' up to '6XL') 
o	Promotional Engineering: The promotion-ids column (49,153 nulls) was engineered into a binary promotion feature, indicating simply whether an order utilized a promotion (1) or not (0).
Following the cleaning phase, the finalized dataset was exported and programmatically uploaded to a PostgreSQL database via a SQLAlchemy engine, establishing a centralized and query-ready data source for the subsequent EDA and modeling phases.
III. EXPLORATORY DATA ANALYSIS (EDA)
The Exploratory Data Analysis was conducted to understand the underlying distribution of the data, uncover hidden patterns, and identify the most significant features driving order cancellations.
1.	Target Variable Distribution & Baseline Metrics 

Dataset holds a baseline success rate of 84.0%. General summary statistics established the average order amount at 646.10 INR, with B2B customers placing slightly higher average orders (697.86 INR) compared to regular customers (645.75 INR). Dataset contains 9 different categories, 69 states and 8922 cities with the Maharashtra placing over 20000 orders. Sales revenue experienced an initial rise in April, followed by a gradual decline through June before the dataset's recording period concluded
2.	Feature Correlation and Importance 

To determine which variables most heavily influence order success, both Pearson Correlation (for linear relationships) and Mutual Information (MI) scores (for non-linear relationships) were calculated. We also train various models with different combination of features based on insights not only from those charts, but also countless pandas queries.
 
 
•	The Dominance of Promotions: The promotion feature showed the strongest linear correlation with the target variable, followed closely by fulfillment_binary (Amazon vs. Merchant) and ship_premium.
•	Non-Linear Drivers: Mutual Information scoring revealed that the Amount feature, despite having a low linear correlation, contains highly significant non-linear predictive power regarding order status. Furthermore, categorical MI analysis highlighted Courier Status as a massive indicator of the final outcome.
3.	Uncovering Critical Business Segments 
By applying K-Means clustering (optimized at k=3 using the Elbow Method) and performing targeted probability analysis, distinct customer segments emerged that dictate the likelihood of an order's success:
•	Overly High Success: Orders fulfilled by Amazon that utilize a promotion are virtually guaranteed to succeed, boasting a 99.37% success rate (only a 0.63% cancellation rate).
•	The "Danger Zone": Conversely, a critical vulnerability was discovered in orders fulfilled by Merchants without any promotional codes attached. While representing only 5.39% of total volume, this specific segment suffers a staggering 99.42% cancellation rate. This singular insight provides immediate strategic value, indicating that Merchant-fulfilled orders require promotional incentives to secure buyer commitment.
•	We suspect this is the data might have been imbalanced and will research further down the report paper.
4.	Data Dimensionality and Collinearity 
During manual data exploration, perfect collinearity was discovered between the fulfilled_by and ship_service_level columns. When an order is fulfilled by a Merchant, the service level is always labeled "Easy Ship," whereas Amazon fulfillments yield null values in that specific column. Recognizing this redundancy allowed for better feature selection, preventing multicollinearity from artificially inflating the Random Forest model's variance.
5.	Temporal and Geographic Trends 
Visualizations detailing these trends are provided in the appendix/dashboard. Broader business trends were visualized, indicating that Maharashtra drives the highest volume of orders by a significant margin. Additionally, a temporal analysis revealed a sharp decline in total sales revenue moving from April into June 2022, signaling potential seasonal shifts or platform-wide issues that warrant future business investigation.
6.	Feature Selection and Refinement 
Based on the insights derived from the EDA, a rigorous feature selection process was applied to prevent data leakage and multicollinearity. The promotion feature was excluded due to target leakage, and ship_service_level was dropped due to perfect collinearity with fulfillment methods. Furthermore, Qty was removed as its near-zero variance provided negligible predictive value.
The first of many iterations achieved very high score:
 
However, upon deploying and actual testing. The model performed with extreme probabilities. Using SHAP and FEATURE IMPORTANCES – we uncovered the imbalance of the dataset is caused by the promotion.
 
Upon various pandas queries, we concluded the data is skewed or misgathered since the beginning. The key findings are:
Orders fullfilled by merchant with no promotion takes 5.39% of the dataset, with an overwhelming 99.42% cancellation rate. On the other hand, those fufilled by Amazon with promotion have a stark contrast of 0.63% cancellation rate. Furthermore, despite having 61.73% of orders have promotion ids, only 3.09% accounted for cancelled. A likely hypothesis is that customers returned the item after trying it on – the promotion_id was used, for the orders that cancelled before deliveries, the system did not keep track of the promotion ids. 
We concluded that the promotion column has potential for data leakage, therefore did not make it in the final training of the model.
The model was trained on 5 features: Amount, fulfillment_binary, B2B_binary, category, and size_ordinal.
The final score being:  With the Feature importances as follow:
Amount: 0.343264
Fulfillment_binary: 0.325385
Category: 0.140645
Size_ordinal: 0.124861
B2B: 0.065845

We believe once the business is expanding, the B2B feature will hold more value and worth keeping. 


    
    Your rules:
    - Answer questions strictly based on the KNOWLEDGE BASE provided above. 
    - Answer questions related to this e-commerce data, Random Forest models, data cleaning, and Exploratory Data Analysis.
    - If a user asks about unrelated topics (e.g., coding help, general history, weather), politely decline and state that you can only answer questions about the order cancellation project.
    - Keep answers concise, professional, and educational.
    - If user asked for charts and specific score, tell them what you know if there is something missing, ask them to go to the EDA tab.
    """
    
    # Initialize the model with the system instructions
    model = genai.GenerativeModel('gemini-2.5-flash', system_instruction=project_rules)
    
    # 3. Initialize chat memory in Streamlit's session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
        
        # Add a friendly greeting from the bot
        st.session_state.messages.append({
            "role": "assistant", 
            "content": "Hello! I am the AI assistant for this e-commerce prediction project. What would you like to know about this project?"
        })

    # 4. Display the chat history on the screen
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # 5. Handle new user input
    prompt = st.chat_input("Ask about the model or data...")
    if prompt:
        
        # Display the user's message immediately
        with st.chat_message("user"):
            st.markdown(prompt)
            
        # Add user message to memory
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Generate the bot's response
        with st.chat_message("assistant"):
            # We need to pass the previous messages to Gemini so it has context
            # We convert Streamlit's dictionary format into a string format Gemini easily reads
            chat_history = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.messages])
            
            response = model.generate_content(chat_history)
            st.markdown(response.text)
            
        # Add bot's response to memory
        st.session_state.messages.append({"role": "assistant", "content": response.text})

# ----------------------------------------------------------------------
# TAB 4: BROWSE SẢN PHẨM (CLOTHING)
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
    # Ẩn điều hướng sub-tab khỏi người dùng (chỉ điều khiển bằng state)
    browser_subtab = st.session_state.browser_subtab

    selected_clothing_type = "Tất cả quần áo"

    selected_product = st.session_state.get("selected_product")

    # 1) VIEW: DỰ ĐOÁN (ẩn bộ lọc, full width)
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
                    st.image(p["image_url"], use_column_width=True)
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
                locked_category=True,
                category_value=selected_product.get("category"),
            )
        st.stop()

    # 2) VIEW: DUYỆT SẢN PHẨM (hiện bộ lọc + danh sách)
    col_left, col_right = st.columns([0.9, 2.1])

    with col_left:
        with st.container(border=True):
            st.markdown("#### 🔍 Bộ lọc sản phẩm (demo)")

            st.markdown("**Loại quần áo**")
            st.radio(
                "",
                ["Tất cả", "Áo thun", "Áo sơ mi", "Áo khoác / Hoodie", "Quần dài", "Quần short", "Đầm / Váy"],
                index=0,
            )

            st.markdown("---")
            st.markdown("**Dành cho**")
            st.checkbox("Nam", value=True)
            st.checkbox("Nữ", value=True)
            st.checkbox("Trẻ em", value=False)

            st.markdown("---")
            st.markdown("**Kích cỡ**")
            size_cols = st.columns(3)
            for i, size in enumerate(["XS", "S", "M", "L", "XL", "2XL"]):
                with size_cols[i % 3]:
                    st.checkbox(size, value=(size in ["M", "L"]))

            st.markdown("---")
            st.markdown("**Khoảng giá (INR)**")
            st.slider("Chọn khoảng giá", min_value=100, max_value=10000, value=(500, 3000), step=100)

            st.markdown("---")
            st.checkbox("Chỉ hiển thị sản phẩm Premium", value=False)
            st.checkbox("Chỉ hiển thị sản phẩm đang giảm giá", value=False)

            st.button("Áp dụng bộ lọc (demo)", use_container_width=True)

    with col_right:
        # Thanh tìm kiếm (thật, nhưng chưa gắn logic lọc)
        with st.container(border=True):
            search_query = st.text_input(
                "Tìm kiếm quần áo",
                placeholder="Ví dụ: hoodie đen, áo thun oversize...",
            )

        st.write("")

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
                "category": "Top",
                "delivery_text": "Dự kiến giao: 3–5 ngày làm việc.",
                "image_url": "https://images.unsplash.com/photo-1521572163474-6864f9cf17ab?auto=format&fit=crop&w=400&q=80",
            },
            {
                "id": "hoodie_black_1299",
                "name": "Hoodie oversize unisex nỉ dày",
                "subtitle": "Đen / Form rộng",
                "price_inr": 1299,
                "category": "Top",
                "delivery_text": "Dự kiến giao: 4–6 ngày làm việc.",
                "image_url": "https://images.unsplash.com/photo-1529927066849-66e1abc70a2e?auto=format&fit=crop&w=400&q=80",
            },
            {
                "id": "shirt_navy_899",
                "name": "Áo bộ dài slim fit",
                "subtitle": "Xanh navy",
                "price_inr": 899,
                "category": "Set",
                "delivery_text": "Dự kiến giao: 2–4 ngày làm việc.",
                "image_url": "https://images.unsplash.com/photo-1528701800489-20be3c30c1d5?auto=format&fit=crop&w=400&q=80",
            },
            {
                "id": "jeans_2500",
                "name": "Quần jean nam dáng slim",
                "subtitle": "Xanh đậm / Co giãn nhẹ",
                "price_inr": 2500,
                "category": "Bottom",
                "delivery_text": "Dự kiến giao: 3–5 ngày làm việc.",
                "image_url": "https://images.unsplash.com/photo-1542272604-787c3835535d?auto=format&fit=crop&w=400&q=80",
            },
            {
                "id": "tee_pink_749",
                "name": "Áo Blouse nữ oversize graphic",
                "subtitle": "Hồng pastel",
                "price_inr": 749,
                "category": "Blouse",
                "delivery_text": "Dự kiến giao: 5–7 ngày làm việc.",
                "image_url": "https://images.unsplash.com/photo-1521572267360-ee0c2909d518?auto=format&fit=crop&w=400&q=80",
            },
        ]

        cols = st.columns(3)
        for idx, p in enumerate(products_demo):
            with cols[idx % 3]:
                with st.container(border=True):
                    st.image(p["image_url"], use_column_width=True)
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
