import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import gc

# --- Load dữ liệu & mô hình ---
@st.cache_data
def load_data():
    base_path = os.path.dirname(__file__)
    data_path = os.path.join(base_path, "data", "rfm_data.pkl")
    df = joblib.load(data_path)
    if not df.index.dtype == "object":
        df.index = df.index.astype(str)
    gc.collect()
    return df


@st.cache_resource
def load_models():
    base_path = os.path.dirname(__file__)
    scaler = joblib.load(os.path.join(base_path, "models", "rfm_scaler.pkl"))
    model = joblib.load(os.path.join(base_path, "models", "kmeans_model.pkl"))
    gc.collect()
    return scaler, model


# --- Ánh xạ nhãn cụm ---
def interpret_cluster(cluster_id):
    return {
        0: "TRUNG BÌNH / KHÁCH PHỔ THÔNG",
        1: "CHURN / KHÁCH RỜI BỎ",
        2: "VIP / KHÁCH GIÁ TRỊ CAO",
    }.get(cluster_id, "Không xác định")


# --- Cấu hình trang ---
st.set_page_config(
    page_title="Phân cụm khách hàng", layout="wide", initial_sidebar_state="expanded"
)

# --- CSS Tùy chỉnh ---
st.markdown(
    """
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 600;
        color: #1E88E5;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.8rem;
        font-weight: 500;
        color: #333;
        margin-top: 1.5rem;
    }
    .card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        margin-bottom: 1rem;
    }
    .metric-container {
        background-color: #e3f2fd;
        border-left: 5px solid #1E88E5;
        padding: 1rem;
        border-radius: 5px;
        margin-bottom: 1rem;
    }
    .sidebar-content {
        padding: 1.5rem 1rem;
    }
    .footer {
        text-align: center;
        margin-top: 2rem;
        padding-top: 1rem;
        border-top: 1px solid #e0e0e0;
        font-size: 0.8rem;
    }
    .info-box {
        background-color: #e8f5e9;
        padding: 1rem;
        border-radius: 5px;
        border-left: 5px solid #43a047;
        margin-bottom: 1rem;
    }
    .warning-box {
        background-color: #fff3e0;
        padding: 1rem;
        border-radius: 5px;
        border-left: 5px solid #ff9800;
        margin-bottom: 1rem;
    }
</style>
""",
    unsafe_allow_html=True,
)


# --- Sidebar ---
with st.sidebar:
    st.markdown('<div class="sidebar-content">', unsafe_allow_html=True)

    st.markdown(
        """
        <h2 style="text-align:center;font-size: 30px;font-weight: bold;">🎓 Đồ án tốt nghiệp<br>Data Science</h2>
        
        <div style="margin-top:1.5rem;">
        <strong>👤 Người thực hiện:</strong><br>  
        <span style="font-size:16px; margin-left:10px;">📌 Nguyễn Quyết Giang Sơn</span><br>   
        <span style="font-size:16px; margin-left:10px;">📌 Phùng Anh Thư</span>
        </div>

        <div style="margin-top:1rem;">
        <strong>👨‍🏫 Giảng viên hướng dẫn:</strong><br>  
        <span style="font-size:16px; margin-left:10px;">🧑‍🏫 Cô Khuất Thùy Phương</span>
        </div>
        
        <div style="margin-top:1rem;">
        <strong>📁 Source code:</strong><br>
        <a href="https://github.com/GiangSon-5/gui_kmeans">
            <span style="font-size:16px; margin-left:10px;">🔗 GitHub Repository</span>
        </a>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<hr>", unsafe_allow_html=True)

    # --- Sidebar: Mô tả và chọn cách nhập ---
    st.markdown(
        """
        <h3>📥 Cách nhập thông tin khách hàng</h3>
        <p>Vui lòng chọn 1 trong 3 cách để thực hiện phân cụm:</p>
        <ul>
            <li><strong>📇 Nhập mã khách hàng:</strong> Chọn khách hàng đã có trong dữ liệu.</li>
            <li><strong>🎛️ Dùng thanh kéo:</strong> Tự nhập chỉ số RFM cho tối đa 5 khách hàng.</li>
            <li><strong>📁 Tải file .csv:</strong> Tải file dữ liệu RFM để phân cụm hàng loạt.</li>
        </ul>
        """,
        unsafe_allow_html=True,
    )

    input_method = st.radio(
        "🔎 Phương thức nhập thông tin:",
        ["📇 Nhập mã khách hàng", "🎛️ Dùng thanh kéo", "📁 Tải file .csv"],
    )

    st.markdown("<hr>", unsafe_allow_html=True)

    # --- Thông tin về RFM ---
    with st.expander("📚 Giải thích chỉ số RFM", expanded=False):
        st.markdown(
            """
            <p><strong>R - Recency (Gần đây):</strong> Số ngày kể từ lần mua gần nhất.</p>
            <p><strong>F - Frequency (Tần suất):</strong> Số lần mua hàng.</p>
            <p><strong>M - Monetary (Giá trị):</strong> Tổng số tiền đã chi tiêu.</p>
            
            <div class="info-box" style="background-color:#f5f5f5; padding:1rem; border-radius:5px; border-left:5px solid #333;">
                <strong style="color:#000; font-weight:bold;">💡 Các cụm khách hàng:</strong>
                <ul style="color:#000; font-weight:bold;">
                    <li><strong style="color:#000; font-weight:bold;">Cụm 0:</strong> Khách hàng trung bình/phổ thông</li>
                    <li><strong style="color:#000; font-weight:bold;">Cụm 1:</strong> Churn/khách rời bỏ</li>
                    <li><strong style="color:#000; font-weight:bold;">Cụm 2:</strong> VIP/khách giá trị cao</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)


# --- Header chính ---
st.markdown(
    '<h1 class="main-header">🔍 Phân cụm khách hàng bằng mô hình RFM + KMeans</h1>',
    unsafe_allow_html=True,
)

# --- Tabs chính ---
tab1, tab2 = st.tabs(["📊 Thực hiện phân cụm", "📚 Giới thiệu & Quy trình"])


# --- Tab 1: Thực hiện phân cụm ---
with tab1:
    df_rfm = load_data()
    scaler, model = load_models()

    # --- 1. Nhập mã khách hàng ---
    if input_method == "📇 Nhập mã khách hàng":
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(
            '<h3 class="sub-header">📇 Tìm kiếm theo mã khách hàng</h3>',
            unsafe_allow_html=True,
        )

        col1, col2 = st.columns([1, 2])

        with col1:
            customer_id = st.text_input("🆔 Nhập mã khách hàng", value="1808")
            search_btn = st.button("🔍 Tìm kiếm", use_container_width=True)

        with col2:
            st.markdown(
                """
                <div class="warning-box" style="background-color: #fff3e0; padding: 1rem; border-radius: 10px; color: #000; font-weight: bold;">
                    💡 <strong>Lưu ý:</strong> Hãy nhập mã khách hàng đã có trong hệ thống để xem thông tin.<br>
                </div>
            """,
                unsafe_allow_html=True,
            )

            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

        if customer_id in df_rfm.index:
            try:
                rfm_row = df_rfm.loc[[customer_id]][["Recency", "Frequency", "Monetary"]]
                scaled_input = scaler.transform(rfm_row)
                cluster_label = model.predict(scaled_input)[0]
                cluster_name = interpret_cluster(cluster_label)

                # Hiển thị thông tin theo khối
                col_info, col_hist = st.columns([3, 2])

                with col_info:
                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    st.markdown(
                        f'<h3 class="sub-header">👤 Thông tin khách hàng {customer_id}</h3>',
                        unsafe_allow_html=True,
                    )

                    # Chia layout cho các chỉ số RFM
                    metric_cols = st.columns(3)
                    with metric_cols[0]:
                        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                        st.metric("📅 Recency", f"{rfm_row['Recency'].values[0]} ngày")
                        st.markdown("</div>", unsafe_allow_html=True)

                    with metric_cols[1]:
                        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                        st.metric("🔁 Frequency", f"{rfm_row['Frequency'].values[0]} lần")
                        st.markdown("</div>", unsafe_allow_html=True)

                    with metric_cols[2]:
                        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                        st.metric("💰 Monetary", f"{rfm_row['Monetary'].values[0]:,.0f} đ")
                        st.markdown("</div>", unsafe_allow_html=True)

                    st.markdown('<div style="margin-top:1rem">', unsafe_allow_html=True)
                    st.metric("🎯 Cụm khách hàng", f"Cụm {cluster_label}", cluster_name)

                    # Chính sách theo cụm
                    policy_color = "#e3f2fd"
                    policy_border = "#1E88E5"
                    policy_text = "Tiềm năng để upsell"

                    if cluster_label == 1:
                        policy_color = "#ffebee"
                        policy_border = "#e53935"
                        policy_text = "Cần remarketing"
                    elif cluster_label == 2:
                        policy_color = "#e8f5e9"
                        policy_border = "#43a047"
                        policy_text = "Giữ chân khách hàng VIP bằng loyalty"

                    st.markdown(
                        f"""
                        <div style="background-color:{policy_color}; padding:1rem; 
                            border-radius:5px; border-left:5px solid {policy_border};">
                            <strong style="color:#000; font-weight:bold;">📌 Chính sách:</strong> <span style="color:#000; font-weight:bold;">{policy_text}</span>
                        </div>
                    """,
                        unsafe_allow_html=True,
                    )
                    st.markdown("</div>", unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)

                gc.collect()  

                with col_hist:
                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    st.markdown(
                        '<h3 class="sub-header">📝 Biểu đồ RFM</h3>',
                        unsafe_allow_html=True,
                    )

                    st.markdown(
                        f"""
                        <div style="margin-bottom:1rem;">
                            <strong>📅 Recency:</strong> {rfm_row['Recency'].values[0]} ngày<br>
                            <strong>🔁 Frequency:</strong> {rfm_row['Frequency'].values[0]} lần<br>
                            <strong>💰 Monetary:</strong> {rfm_row['Monetary'].values[0]:,.0f} đ
                        </div>
                    """,
                        unsafe_allow_html=True,
                    )

                    # Biểu đồ kết hợp (Recency, Frequency, Monetary)
                    fig, ax = plt.subplots()
                    labels = ["Recency", "Frequency", "Monetary"]
                    values = [
                        rfm_row["Recency"].values[0],
                        rfm_row["Frequency"].values[0],
                        rfm_row["Monetary"].values[0],
                    ]
                    colors = ["#1E88E5", "#ff7043", "#43a047"]

                    ax.bar(labels, values, color=colors)
                    ax.set_title("Biểu đồ RFM")
                    ax.set_ylabel("Giá trị")
                    st.pyplot(fig)
                    plt.close(fig)  
                    gc.collect()    

                    st.markdown("</div>", unsafe_allow_html=True)

                gc.collect()  

                st.toast(f"✅ Khách hàng {customer_id} thuộc cụm **{cluster_label} – {cluster_name}**")

            except Exception as e:
                st.error(f"❌ Không thể phân cụm: {str(e)}")
        else:
            if customer_id:
                st.error("⚠️ Mã khách hàng không tồn tại trong dữ liệu.")

    # --- 2. Dùng slider ---
    elif input_method == "🎛️ Dùng thanh kéo":
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(
            '<h3 class="sub-header">🎛️ Nhập thông tin khách hàng mới</h3>',
            unsafe_allow_html=True,
        )

        tab_customers = st.tabs(
            ["Khách hàng 1", "Khách hàng 2", "Khách hàng 3", "Khách hàng 4", "Khách hàng 5"]
        )
        customer_data = []

        for i, tab in enumerate(tab_customers):
            with tab:
                st.markdown(f"#### 👤 Khách hàng {i+1}")
                col1, col2 = st.columns(2)

                with col1:
                    r = st.slider(f"🕒 Recency (ngày)", 1, 365, 100, key=f"recency_{i}")
                    f = st.slider(f"🔁 Frequency (lần)", 1, 50, 5, key=f"frequency_{i}")

                with col2:
                    m = st.slider(
                        f"💰 Monetary (nghìn đồng)", 1, 1000, 100, key=f"monetary_{i}"
                    )

                customer_data.append([r, f, m])

        st.markdown("</div>", unsafe_allow_html=True)

        if st.button("🎯 Phân cụm khách hàng", use_container_width=True, type="primary"):
            df_customer = pd.DataFrame(
                customer_data, columns=["Recency", "Frequency", "Monetary"]
            )

            try:
                scaled_input = scaler.transform(df_customer)
                clusters = model.predict(scaled_input)
                df_customer["Cụm"] = clusters
                df_customer["Phân nhóm"] = df_customer["Cụm"].apply(interpret_cluster)

                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown(
                    '<h3 class="sub-header">📊 Kết quả phân cụm</h3>',
                    unsafe_allow_html=True,
                )

                result_df = pd.DataFrame(
                    {
                        "Khách hàng": [f"Khách hàng {i+1}" for i in range(len(df_customer))],
                        "Recency": df_customer["Recency"],
                        "Frequency": df_customer["Frequency"],
                        "Monetary": df_customer["Monetary"],
                        "Cụm": df_customer["Cụm"],
                        "Phân nhóm": df_customer["Phân nhóm"],
                    }
                )

                st.dataframe(
                    result_df,
                    use_container_width=True,
                    column_config={
                        "Khách hàng": st.column_config.TextColumn("Khách hàng", width="medium"),
                        "Recency": st.column_config.NumberColumn("Recency (ngày)", format="%d"),
                        "Frequency": st.column_config.NumberColumn("Frequency (lần)", format="%d"),
                        "Monetary": st.column_config.NumberColumn("Monetary (nghìn đ)", format="%d"),
                        "Cụm": st.column_config.NumberColumn("Cụm", format="%d"),
                        "Phân nhóm": st.column_config.TextColumn("Phân nhóm", width="large"),
                    },
                )
                st.markdown("</div>", unsafe_allow_html=True)

                for i, row in df_customer.iterrows():
                    with st.expander(f"👤 Chi tiết khách hàng {i+1} (Cụm {row['Cụm']})", expanded=False):
                        col_detail, col_policy = st.columns([3, 2])

                        with col_detail:
                            metric_cols = st.columns(3)
                            with metric_cols[0]:
                                st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                                st.metric("📅 Recency", f"{row['Recency']} ngày")
                                st.markdown("</div>", unsafe_allow_html=True)

                            with metric_cols[1]:
                                st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                                st.metric("🔁 Frequency", f"{row['Frequency']} lần")
                                st.markdown("</div>", unsafe_allow_html=True)

                            with metric_cols[2]:
                                st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                                st.metric("💰 Monetary", f"{row['Monetary']:,.0f} đ")
                                st.markdown("</div>", unsafe_allow_html=True)

                        with col_policy:
                            policy_color = "#e3f2fd"
                            policy_border = "#1E88E5"
                            policy_text = "Tiềm năng để upsell"

                            if row["Cụm"] == 1:
                                policy_color = "#ffebee"
                                policy_border = "#e53935"
                                policy_text = "Cần remarketing"
                            elif row["Cụm"] == 2:
                                policy_color = "#e8f5e9"
                                policy_border = "#43a047"
                                policy_text = "Giữ chân khách hàng VIP bằng loyalty"

                            st.markdown(
                                f"""
                                <div style="background-color:{policy_color}; padding:1rem; 
                                    border-radius:5px; border-left:5px solid {policy_border}; color:black;">
                                <strong>📌 Chính sách:</strong> {policy_text}
                                </div>
                            """,
                                unsafe_allow_html=True,
                            )

                            st.metric("🎯 Cụm khách hàng", f"Cụm {row['Cụm']}", row["Phân nhóm"])

                        # Biểu đồ RFM
                        st.subheader("📊 Biểu đồ RFM")
                        fig, ax = plt.subplots(figsize=(16, 4))
                        labels = ["Recency", "Frequency", "Monetary"]
                        values = [row["Recency"], row["Frequency"], row["Monetary"]]
                        colors = ["#1E88E5", "#ff7043", "#43a047"]
                        ax.bar(labels, values, color=colors)
                        ax.set_title(f"Biểu đồ RFM Khách hàng {i+1}")
                        ax.set_ylabel("Giá trị")

                        st.pyplot(fig)
                        plt.close(fig)     # 🔒 Đóng biểu đồ sau khi dùng
                        gc.collect()       # 🧹 Thu dọn bộ nhớ sau mỗi biểu đồ

            except Exception as e:
                st.error(f"❌ Lỗi khi phân cụm: {str(e)}")


                st.error(f"❌ Lỗi khi phân cụm: {str(e)}")

    # --- 3. Upload file .csv ---
    elif input_method == "📁 Tải file .csv":
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(
            '<h3 class="sub-header">📂 Tải file dữ liệu khách hàng</h3>',
            unsafe_allow_html=True,
        )

        col_upload, col_info = st.columns([2, 3])

        with col_upload:
            uploaded_file = st.file_uploader("📤 Chọn file CSV", type=["csv"])

        with col_info:
            st.markdown(
                """ 
                <div class="info-box" style="background-color:#f5f5f5; padding:1rem; 
                    border-radius:5px; border-left:5px solid #333;">
                    <strong style="color:#000; font-weight:bold;">📋 Yêu cầu file CSV:</strong>
                    <ul style="color:#000; font-weight:bold;">
                        <li><code>Recency</code>: Số ngày kể từ lần mua gần nhất</li>
                        <li><code>Frequency</code>: Số lần mua hàng</li>
                        <li><code>Monetary</code>: Tổng số tiền đã chi tiêu</li>
                        <li><code>CustomerID</code>: Mã khách hàng (không bắt buộc)</li>
                    </ul>
                </div>
            """,
                unsafe_allow_html=True,
            )

            # Tải file mẫu
            st.markdown(
                """
                <a href="https://github.com/GiangSon-5/gui_kmeans/blob/main/data/sample_rfm_input.csv" 
                   style="text-decoration:none;">
                    <div style="background-color:#f5f5f5; padding:0.8rem; 
                        border-radius:5px; text-align:center; cursor:pointer;">
                        📥 Tải file mẫu CSV
                    </div>
                </a>
            """,
                unsafe_allow_html=True,
            )

        st.markdown("</div>", unsafe_allow_html=True)

        if uploaded_file is not None:
            try:
                df_uploaded = pd.read_csv(uploaded_file)
                required_cols = {"Recency", "Frequency", "Monetary"}

                if not required_cols.issubset(df_uploaded.columns):
                    st.error("❌ File chưa có đủ các cột cần thiết!")
                else:
                    df_input = df_uploaded[["Recency", "Frequency", "Monetary"]]
                    scaled_input = scaler.transform(df_input)
                    clusters = model.predict(scaled_input)
                    df_uploaded["Cụm"] = clusters
                    df_uploaded["Phân nhóm"] = df_uploaded["Cụm"].apply(
                        interpret_cluster
                    )

                    st.success("✅ Phân cụm thành công!")

                    # Thống kê kết quả phân cụm
                    cluster_stats = df_uploaded["Cụm"].value_counts().reset_index()
                    cluster_stats.columns = ["Cụm", "Số lượng"]

                    col_table, col_chart = st.columns([3, 2])

                    with col_table:
                        st.markdown('<div class="card">', unsafe_allow_html=True)
                        st.markdown(
                            '<h3 class="sub-header">📊 Kết quả phân cụm</h3>',
                            unsafe_allow_html=True,
                        )

                        # Định dạng bảng dữ liệu
                        st.dataframe(
                            df_uploaded,
                            use_container_width=True,
                            column_config={
                                "CustomerID": st.column_config.TextColumn(
                                    "Mã KH", width="small"
                                ),
                                "Recency": st.column_config.NumberColumn(
                                    "Recency", format="%d"
                                ),
                                "Frequency": st.column_config.NumberColumn(
                                    "Frequency", format="%d"
                                ),
                                "Monetary": st.column_config.NumberColumn(
                                    "Monetary", format="%d"
                                ),
                                "Cụm": st.column_config.NumberColumn(
                                    "Cụm", format="%d"
                                ),
                                "Phân nhóm": st.column_config.TextColumn(
                                    "Phân nhóm", width="large"
                                ),
                            },
                        )
                        st.markdown("</div>", unsafe_allow_html=True)

                    with col_chart:
                        st.markdown('<div class="card">', unsafe_allow_html=True)
                        st.markdown(
                            '<h3 class="sub-header">📈 Phân bố cụm</h3>',
                            unsafe_allow_html=True,
                        )

                        # Biểu đồ phân bố cụm
                        fig, ax = plt.subplots()
                        colors = ["#1976D2", "#E53935", "#43A047"]
                        labels = [
                            "Trung bình/Phổ thông",
                            "Churn/Rời bỏ",
                            "VIP/Giá trị cao",
                        ]

                        # Tạo biểu đồ hình tròn
                        wedges, texts, autotexts = ax.pie(
                            cluster_stats["Số lượng"],
                            labels=None,
                            autopct="%1.1f%%",
                            startangle=90,
                            colors=colors,
                        )

                        # Thêm chú thích
                        ax.legend(
                            wedges,
                            [f"Cụm {i}: {label}" for i, label in enumerate(labels)],
                            loc="center left",
                            bbox_to_anchor=(1, 0, 0.5, 1),
                        )

                        plt.setp(autotexts, size=10, weight="bold")
                        ax.set_title("Phân bố khách hàng theo cụm")

                        st.pyplot(fig)
                        st.markdown("</div>", unsafe_allow_html=True)

                    # Chi tiết từng khách hàng
                    st.markdown(
                        '<h3 class="sub-header">👥 Chi tiết khách hàng theo cụm</h3>',
                        unsafe_allow_html=True,
                    )

                    # Tạo tab cho từng cụm
                    tab_clusters = st.tabs(
                        [
                            f"Cụm 0: Trung bình ({(df_uploaded['Cụm'] == 0).sum()} khách hàng)",
                            f"Cụm 1: Churn ({(df_uploaded['Cụm'] == 1).sum()} khách hàng)",
                            f"Cụm 2: VIP ({(df_uploaded['Cụm'] == 2).sum()} khách hàng)",
                        ]
                    )

                    # Hiển thị chi tiết từng cụm
                    for cluster_id, tab in enumerate(tab_clusters):
                        with tab:
                            cluster_df = df_uploaded[df_uploaded["Cụm"] == cluster_id]

                            if len(cluster_df) > 0:
                                # Hiển thị thông tin thống kê cụm
                                st.markdown(
                                    '<div class="card">', unsafe_allow_html=True
                                )

                                stats_cols = st.columns(3)
                                with stats_cols[0]:
                                    avg_recency = cluster_df["Recency"].mean()
                                    st.metric(
                                        "📅 Trung bình Recency",
                                        f"{avg_recency:.1f} ngày",
                                    )

                                with stats_cols[1]:
                                    avg_frequency = cluster_df["Frequency"].mean()
                                    st.metric(
                                        "🔁 Trung bình Frequency",
                                        f"{avg_frequency:.1f} lần",
                                    )

                                with stats_cols[2]:
                                    avg_monetary = cluster_df["Monetary"].mean()
                                    st.metric(
                                        "💰 Trung bình Monetary",
                                        f"{avg_monetary:.1f} đ",
                                    )

                                st.markdown("</div>", unsafe_allow_html=True)

                                # Chính sách cho cụm này
                                policy_text = "Tiềm năng để upsell"
                                if cluster_id == 1:
                                    policy_text = "Cần remarketing"
                                elif cluster_id == 2:
                                    policy_text = "Giữ chân khách hàng VIP bằng loyalty"

                                st.info(
                                    f"**📌 Chính sách cho cụm {cluster_id}:** {policy_text}"
                                )

                                # Danh sách khách hàng trong cụm
                                st.dataframe(
                                    cluster_df,
                                    use_container_width=True,
                                    column_config={
                                        "CustomerID": st.column_config.TextColumn(
                                            "Mã KH", width="small"
                                        ),
                                        "Recency": st.column_config.NumberColumn(
                                            "Recency", format="%d"
                                        ),
                                        "Frequency": st.column_config.NumberColumn(
                                            "Frequency", format="%d"
                                        ),
                                        "Monetary": st.column_config.NumberColumn(
                                            "Monetary", format="%d"
                                        ),
                                        "Cụm": None,  # Ẩn cột Cụm
                                        "Phân nhóm": None,  # Ẩn cột Phân nhóm
                                    },
                                )
                            else:
                                st.info(
                                    f"Không có khách hàng nào thuộc cụm {cluster_id}"
                                )

            except Exception as e:
                st.error(f"⚠️ Lỗi xử lý file: {str(e)}")


# --- Tab 2: Giới thiệu & Quy trình ---
with tab2:
    st.markdown(
        '<h2 class="main-header">📚 Giới thiệu về mô hình RFM và K-Means</h2>',
        unsafe_allow_html=True,
    )

    # Chia thành 2 tab con
    subtab1, subtab2 = st.tabs(["🔍 Mô hình RFM", "🧩 Thuật toán K-Means"])

    with subtab1:
        col1, col2 = st.columns([3, 2])

        with col1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown(
                """
                <h3 class="sub-header">🔍 Mô hình RFM là gì?</h3>
                
                <p>RFM là một phương pháp phân tích khách hàng dựa trên 3 chỉ số:</p>
                
                <ul>
                    <li><strong>R - Recency (Gần đây):</strong> Thời gian kể từ lần mua hàng gần nhất</li>
                    <li><strong>F - Frequency (Tần suất):</strong> Số lần khách hàng mua hàng trong một khoảng thời gian</li>
                    <li><strong>M - Monetary (Giá trị):</strong> Tổng số tiền khách hàng đã chi tiêu</li>
                </ul>
                
                <p>Mô hình RFM giúp doanh nghiệp phân loại khách hàng và đưa ra các chiến lược marketing phù hợp với từng nhóm.</p>
            """,
                unsafe_allow_html=True,
            )
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown(
                """
                <h3 class="sub-header">📊 Ứng dụng của mô hình RFM</h3>
                
                <ul>
                    <li><strong>Marketing cá nhân hóa:</strong> Xây dựng chiến lược tiếp thị cho từng phân khúc khách hàng</li>
                    <li><strong>Tối ưu hóa chi phí:</strong> Tập trung nguồn lực vào các khách hàng có giá trị cao</li>
                    <li><strong>Giữ chân khách hàng:</strong> Xác định và tập trung vào khách hàng có nguy cơ rời bỏ</li>
                    <li><strong>Cross-selling & Up-selling:</strong> Gợi ý sản phẩm phù hợp với từng nhóm khách hàng</li>
                </ul>
            """,
                unsafe_allow_html=True,
            )
            st.markdown("</div>", unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown(
                """
                <h3 class="sub-header" ; font-weight: bold;">💎 Giá trị của RFM</h3>
                
                <p  font-weight: bold;">
                    RFM giúp doanh nghiệp hiểu rõ hơn về khách hàng của mình và đưa ra quyết định kinh doanh hiệu quả.
                </p>
                
                <div class="info-box" style="color: #000; font-weight: bold;">
                    <strong>🔑 Lợi ích chính:</strong>
                    <ul>
                        <li>Phân loại khách hàng một cách khách quan</li>
                        <li>Tối ưu hóa chiến lược marketing</li>
                        <li>Tăng tỷ lệ giữ chân khách hàng</li>
                        <li>Nâng cao giá trị vòng đời khách hàng</li>
                    </ul>
                </div>
            """,
                unsafe_allow_html=True,
            )

    with subtab2:
        col1, col2 = st.columns([3, 2])

        with col1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown(
                """
                <h3 class="sub-header">🧩 Thuật toán K-Means là gì?</h3>
                
                <p>K-Means là một thuật toán phân cụm (clustering) phổ biến trong học máy không giám sát. Thuật toán phân chia dữ liệu thành K cụm dựa trên khoảng cách giữa các điểm dữ liệu.</p>
                
                <p><strong>Nguyên lý hoạt động:</strong></p>
                <ol>
                    <li>Chọn K điểm trung tâm (centroids) ban đầu</li>
                    <li>Gán mỗi điểm dữ liệu vào cụm có tâm gần nhất</li>
                    <li>Tính toán lại tâm của mỗi cụm</li>
                    <li>Lặp lại bước 2 và 3 cho đến khi hội tụ</li>
                </ol>
            """,
                unsafe_allow_html=True,
            )
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown(
                """
                <h3 class="sub-header">🔄 Quy trình áp dụng</h3>
                
                <p>Quy trình phân cụm khách hàng bằng RFM và K-Means:</p>
                
                <ol>
                    <li><strong>Thu thập dữ liệu:</strong> Dữ liệu giao dịch của khách hàng</li>
                    <li><strong>Tính toán chỉ số RFM:</strong> Recency, Frequency, Monetary</li>
                    <li><strong>Chuẩn hóa dữ liệu:</strong> Đưa các chỉ số về cùng một thang đo</li>
                    <li><strong>Áp dụng K-Means:</strong> Phân cụm khách hàng</li>
                    <li><strong>Phân tích kết quả:</strong> Hiểu đặc điểm của từng cụm</li>
                    <li><strong>Xây dựng chiến lược:</strong> Chiến lược marketing cho từng nhóm khách hàng</li>
                </ol>
            """,
                unsafe_allow_html=True,
            )
            st.markdown("</div>", unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown(
                """
                <h3 class="sub-header" ; font-weight: bold;">📈 Lợi ích của thuật toán</h3>
                
                <div class="info-box" style="color: #000; font-weight: bold; background-color: #e3f2fd; padding: 1rem; border-radius: 10px; margin-bottom: 1rem;">
                    <strong>💡 Ưu điểm:</strong>
                    <ul>
                        <li>Đơn giản, dễ hiểu và triển khai</li>
                        <li>Hiệu quả với dữ liệu lớn</li>
                        <li>Tốc độ xử lý nhanh</li>
                        <li>Kết quả dễ diễn giải</li>
                    </ul>
                </div>
                
                <div class="warning-box" style="color: #000; font-weight: bold; background-color: #fff3e0; padding: 1rem; border-radius: 10px;">
                    <strong>⚠️ Lưu ý:</strong>
                    <ul>
                        <li>Cần chọn số cụm K phù hợp</li>
                        <li>Nhạy cảm với điểm dữ liệu ngoại lệ</li>
                        <li>Kết quả phụ thuộc vào điểm khởi tạo</li>
                    </ul>
                </div>
            """,
                unsafe_allow_html=True,
            )

            # Biểu đồ K-Means
            # Xác định đường dẫn đến thư mục chứa ảnh
            base_path = os.path.dirname(__file__)
            image_dir = os.path.join(base_path, "images")

            # Đường dẫn đến file ảnh
            image_file = "kmeans.png"  # Tên file ảnh
            image_path = os.path.join(image_dir, image_file)

            # Kiểm tra nếu file tồn tại thì hiển thị ảnh
            if os.path.exists(image_path):
                st.image(image_path, caption="Minh họa thuật toán K-Means", use_container_width=True)
            else:
                st.warning(f"Không tìm thấy file: `{image_path}`")
            st.markdown("</div>", unsafe_allow_html=True)

    # Thông tin về dự án
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<h3 class="sub-header">🎯 Mục tiêu dự án</h3>', unsafe_allow_html=True)

    goal_cols = st.columns(3)
    with goal_cols[0]:
        st.markdown(
            """
            <div style="text-align:center; padding: 1rem; background-color:#e8f5e9; border-radius:10px; height:100%;">
                <h4 style="color: #000; font-weight: bold;">📊 Phân tích</h4>
                <p style="color: #000; font-weight: bold;">Phân tích hành vi khách hàng dựa trên mô hình RFM để hiểu rõ giá trị và hành vi mua hàng</p>
            </div>
        """,
            unsafe_allow_html=True,
        )

    with goal_cols[1]:
        st.markdown(
            """
            <div style="text-align:center; padding: 1rem; background-color:#e3f2fd; border-radius:10px; height:100%;">
                <h4 style="color: #000; font-weight: bold;">🧩 Phân cụm</h4>
                <p style="color: #000; font-weight: bold;">Áp dụng thuật toán K-Means để phân nhóm khách hàng thành các cụm có đặc điểm tương đồng</p>
            </div>
        """,
            unsafe_allow_html=True,
        )

    with goal_cols[2]:
        st.markdown(
            """
            <div style="text-align:center; padding: 1rem; background-color:#fff3e0; border-radius:10px; height:100%;">
                <h4 style="color: #000; font-weight: bold;">📈 Ứng dụng</h4>
                <p style="color: #000; font-weight: bold;">Xây dựng chiến lược marketing phù hợp cho từng nhóm khách hàng để tối ưu hóa doanh thu</p>
            </div>
        """,
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)


# --- Footer ---
st.markdown('<div class="footer">', unsafe_allow_html=True)
st.markdown(
    """
    © 2025 Đồ án tốt nghiệp Data Science | Trung tâm Tin học
""",
    unsafe_allow_html=True,
)
st.markdown("</div>", unsafe_allow_html=True)

# --- Tab 2: Giới thiệu ---
with tab2:
    st.title("Trung Tâm Tin Học")
    st.subheader(":mortar_board: Đồ án tốt nghiệp Data Science")
    st.header("1. Giới thiệu nội dung dự án")
    st.markdown(
        """
    Dự án nhằm phân khúc khách hàng dựa trên mô hình **RFM (Recency - Frequency - Monetary)**: 

    **Quy trình thực hiện:**
    - Tiến hành EDA để hiểu hành vi khách hàng
    - Tính toán chỉ số RFM
    - Chuẩn hóa dữ liệu & áp dụng KMeans clustering
    - Đánh giá bằng GMM - PCA
    """
    )

    image_files = [
        ("1.EDA_product.png", "EDA: Phân tích sản phẩm"),
        ("2.EDA_sales.png", "EDA: Doanh thu theo ngày"),
        ("3.top_10.eda.png", "Top 10 khách hàng theo doanh thu"),
        ("4.RFM_historgram.png", "Biểu đồ histogram RFM"),
        ("5.ebow_kmeans.png", "Lựa chọn k (elbow method)"),
        ("6.bubble_chart_kmeans.png", "Phân cụm KMeans qua bubble chart"),
        ("7.GMM-PCA.png", "Phân cụm GMM-PCA"),
    ]

    base_path = os.path.dirname(__file__)
    image_dir = os.path.join(base_path, "images")

    for file, caption in image_files:
        image_path = os.path.join(image_dir, file)
        if os.path.exists(image_path):
            st.image(image_path, caption=caption, use_container_width=True)
        else:
            st.warning(f"Không tìm thấy file: `{image_path}`")

    st.markdown(
        """
    Dựa vào RFM, dự án chia thành 3 cụm khách hàng:

    - **Cluster 1: CHURN / KHÁCH RỜI BỎ** – Cần remarketing
    - **Cluster 2: VIP / GIÁ TRỊ CAO** – Giữ chân bằng loyalty
    - **Cluster 0: TRUNG BÌNH / PHỔ THÔNG** – Tiềm năng để upsell
    """
    )
