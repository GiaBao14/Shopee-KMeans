from pathlib import Path
from textwrap import fill

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "data.csv"
OUTPUT_DIR = BASE_DIR / "output"

PRICE_RESULT_PATH = OUTPUT_DIR / "kmeans_price_result.csv"
SEGMENT_RESULT_PATH = OUTPUT_DIR / "kmeans_product_segment_result.csv"
FULL_FEATURE_RESULT_PATH = OUTPUT_DIR / "kmeans_full_feature_result.csv"


st.set_page_config(
    page_title="Dashboard K-Means Sản Phẩm Shopee",
    page_icon="📊",
    layout="wide"
)


@st.cache_data
def load_csv(path):
    return pd.read_csv(path, encoding="utf-8-sig")


def format_price(value):
    return f"{int(value):,} VND".replace(",", ".")


def wrap_label(value, width=16):
    return fill(str(value), width=width)


def display_label(value):
    labels = {
        "Gia thap": "Giá thấp",
        "Gia trung binh": "Giá trung bình",
        "Gia cao": "Giá cao",
        "San pham hot": "Sản phẩm hot",
        "San pham it nguoi mua": "Sản phẩm ít người mua",
        "San pham danh gia cao": "Sản phẩm đánh giá cao"
    }

    return labels.get(str(value), str(value))


def prepare_data(data):
    data = data.copy()

    for column in ["price", "buyer_count", "rating"]:
        if column in data.columns:
            data[column] = pd.to_numeric(
                data[column],
                errors="coerce"
            )

    return data


def show_missing_file(path, command):
    st.warning(
        f"Chưa tìm thấy file `{path.relative_to(BASE_DIR)}`. "
        f"Hãy chạy `{command}` trước để tạo kết quả."
    )


def draw_cluster_count_chart(data, cluster_column, title):
    cluster_count = data[cluster_column].value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(
        cluster_count.index.astype(str),
        cluster_count.values,
        color="#4e79a7",
        edgecolor="white"
    )

    ax.set_title(title, fontweight="bold", pad=12)
    ax.set_xlabel("Cụm")
    ax.set_ylabel("Số lượng sản phẩm")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            str(int(height)),
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold"
        )

    st.pyplot(fig)


def draw_value_count_bar_chart(data, column, title, color="#2f80ed"):
    value_count = data[column].value_counts()

    fig, ax = plt.subplots(figsize=(8, 4.8))
    bars = ax.bar(
        range(len(value_count)),
        value_count.values,
        color=color,
        edgecolor="white"
    )

    ax.set_title(title, fontweight="bold", pad=12)
    ax.set_xlabel("")
    ax.set_ylabel("Số lượng sản phẩm")
    ax.set_xticks(range(len(value_count)))
    ax.set_xticklabels(
        [wrap_label(display_label(label), width=14) for label in value_count.index],
        rotation=0,
        ha="center"
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            str(int(height)),
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold"
        )

    fig.subplots_adjust(bottom=0.25)
    st.pyplot(fig)


def draw_top_price_chart(data):
    top_6 = (
        data.sort_values("price", ascending=False)
        .head(6)
        .reset_index(drop=True)
    )

    fig, ax = plt.subplots(figsize=(11, 5.5))
    bars = ax.bar(
        range(len(top_6)),
        top_6["price"],
        color="#2f80ed",
        edgecolor="white"
    )

    ax.set_title("Top 6 sản phẩm có giá bán cao nhất", fontweight="bold", pad=12)
    ax.set_xlabel("Sản phẩm")
    ax.set_ylabel("Giá bán")
    ax.set_xticks(range(len(top_6)))
    ax.set_xticklabels(
        [wrap_label(name) for name in top_6["product_name"]],
        rotation=0,
        ha="center"
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    max_price = top_6["price"].max()

    for bar, price in zip(bars, top_6["price"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max_price * 0.02,
            format_price(price),
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold"
        )

    ax.set_ylim(0, max_price * 1.18)
    fig.subplots_adjust(bottom=0.28)
    st.pyplot(fig)


def searchable_table(data):
    st.subheader("Bảng dữ liệu sản phẩm")

    search = st.text_input(
        "Tìm kiếm sản phẩm",
        placeholder="Nhập tên sản phẩm, shop hoặc danh mục..."
    )

    filtered = data.copy()

    category_options = ["Tất cả"] + sorted(filtered["category"].dropna().unique())
    selected_category = st.selectbox("Lọc theo danh mục", category_options)

    if selected_category != "Tất cả":
        filtered = filtered[filtered["category"] == selected_category]

    if search:
        keyword = search.strip()
        filtered = filtered[
            filtered["product_name"].str.contains(keyword, case=False, na=False)
            | filtered["shop_name"].str.contains(keyword, case=False, na=False)
            | filtered["category"].str.contains(keyword, case=False, na=False)
        ]

    st.caption(f"Hiển thị {len(filtered)} / {len(data)} sản phẩm")
    st.dataframe(
        filtered,
        use_container_width=True,
        hide_index=True
    )


def main():
    st.title("Dashboard K-Means Sản Phẩm Shopee")
    st.write(
        "Dashboard hiển thị bảng dữ liệu, tìm kiếm sản phẩm và các biểu đồ "
        "phân cụm K-Means theo giá, lượt mua, đánh giá và đầy đủ đặc trưng."
    )

    if not DATA_PATH.exists():
        st.error(f"Không tìm thấy `{DATA_PATH.relative_to(BASE_DIR)}`.")
        return

    dataset = prepare_data(load_csv(DATA_PATH))

    total_products = len(dataset)
    average_price = dataset["price"].mean()
    total_buyers = dataset["buyer_count"].sum()
    average_rating = dataset["rating"].mean()

    metric_cols = st.columns(4)
    metric_cols[0].metric("Số sản phẩm", f"{total_products}")
    metric_cols[1].metric("Giá trung bình", format_price(average_price))
    metric_cols[2].metric("Tổng lượt mua", f"{int(total_buyers):,}".replace(",", "."))
    metric_cols[3].metric("Đánh giá trung bình", f"{average_rating:.2f}")

    dataset_tab, price_tab, segment_tab, full_feature_tab = st.tabs(
        [
            "Dữ liệu & Tìm kiếm",
            "K-Means Giá bán",
            "K-Means Hot/Ít mua/Đánh giá",
            "K-Means Đầy đủ đặc trưng"
        ]
    )

    with dataset_tab:
        searchable_table(dataset)

    with price_tab:
        st.subheader("Phân cụm theo giá bán")

        if PRICE_RESULT_PATH.exists():
            price_df = prepare_data(load_csv(PRICE_RESULT_PATH))

            cols = st.columns([1, 1])

            with cols[0]:
                draw_top_price_chart(price_df)

            with cols[1]:
                if "price_group" in price_df.columns:
                    draw_value_count_bar_chart(
                        price_df,
                        "price_group",
                        "Số lượng sản phẩm theo nhóm giá"
                    )
                elif "price_cluster" in price_df.columns:
                    draw_cluster_count_chart(
                        price_df,
                        "price_cluster",
                        "Số lượng sản phẩm theo cụm giá"
                    )

            st.write("Kết quả phân cụm giá")
            st.dataframe(
                price_df,
                use_container_width=True,
                hide_index=True
            )

        else:
            show_missing_file(
                PRICE_RESULT_PATH,
                "python clustering/kmeans_price_cluster.py"
            )

    with segment_tab:
        st.subheader("Phân cụm sản phẩm hot, ít người mua, đánh giá cao")

        if SEGMENT_RESULT_PATH.exists():
            segment_df = prepare_data(load_csv(SEGMENT_RESULT_PATH))

            cols = st.columns([1, 1])

            with cols[0]:
                if "product_segment" in segment_df.columns:
                    draw_value_count_bar_chart(
                        segment_df,
                        "product_segment",
                        "Số lượng sản phẩm theo nhóm",
                        color="#4e79a7"
                    )

            with cols[1]:
                st.write("Top sản phẩm theo lượt mua")
                top_buyers = (
                    segment_df.sort_values("buyer_count", ascending=False)
                    .head(8)[["product_name", "buyer_count", "rating", "product_segment"]]
                )
                st.dataframe(top_buyers, use_container_width=True, hide_index=True)

            st.write("Kết quả phân cụm sản phẩm")
            st.dataframe(
                segment_df,
                use_container_width=True,
                hide_index=True
            )
        else:
            show_missing_file(
                SEGMENT_RESULT_PATH,
                "python clustering/kmeans_product_segment.py"
            )

    with full_feature_tab:
        st.subheader("K-Means đầy đủ đặc trưng")
        st.caption(
            "Đặc trưng sử dụng: price, buyer_count, rating, category_encoded."
        )

        if FULL_FEATURE_RESULT_PATH.exists():
            full_df = prepare_data(load_csv(FULL_FEATURE_RESULT_PATH))

            cols = st.columns([1, 1])

            with cols[0]:
                draw_cluster_count_chart(
                    full_df,
                    "cluster",
                    "Số lượng sản phẩm theo cụm đầy đủ đặc trưng"
                )

            with cols[1]:
                if "cluster_meaning" in full_df.columns:
                    st.write("Ý nghĩa cụm")
                    meaning_table = (
                        full_df[["cluster", "cluster_meaning"]]
                        .drop_duplicates()
                        .sort_values("cluster")
                    )
                    st.dataframe(
                        meaning_table,
                        use_container_width=True,
                        hide_index=True
                    )

            st.write("Kết quả K-Means đầy đủ đặc trưng")
            st.dataframe(
                full_df,
                use_container_width=True,
                hide_index=True
            )
        else:
            show_missing_file(
                FULL_FEATURE_RESULT_PATH,
                "python clustering/kmeans_full_feature.py"
            )


if __name__ == "__main__":
    main()
