from pathlib import Path
import re
from textwrap import fill
import unicodedata

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "data.csv"
OUTPUT_DIR = BASE_DIR / "output"
IMAGE_DIR = BASE_DIR / "data" / "images"
PRODUCT_IMAGE_DIR = IMAGE_DIR / "products"
PLACEHOLDER_IMAGE_DIR = IMAGE_DIR / "placeholders"

PRICE_RESULT_PATH = OUTPUT_DIR / "kmeans_price_result.csv"
SEGMENT_RESULT_PATH = OUTPUT_DIR / "kmeans_product_segment_result.csv"
FULL_FEATURE_RESULT_PATH = OUTPUT_DIR / "kmeans_full_feature_result.csv"


st.set_page_config(
    page_title="Dashboard K-Means Sản Phẩm Shopee",
    page_icon="📊",
    layout="wide"
)


@st.cache_data
def load_csv(path, modified_time):
    return pd.read_csv(path, encoding="utf-8-sig")


def read_csv(path):
    return load_csv(path, path.stat().st_mtime)


def format_price(value):
    return f"{int(value):,} VND".replace(",", ".")


def wrap_label(value, width=16):
    return fill(str(value), width=width)


def slugify(value):
    text = str(value).replace("Đ", "D").replace("đ", "d")
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^a-zA-Z0-9]+", "-", text).strip("-").lower()
    return text or "unknown"


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


def load_result_if_exists(path):
    if path.exists():
        return prepare_data(read_csv(path))

    return pd.DataFrame()


def find_local_image(product):
    product_name = product.get("product_name", "")
    category = product.get("category", "")

    if "image_url" in product and pd.notna(product["image_url"]):
        image_url = str(product["image_url"]).strip()

        if image_url.startswith(("http://", "https://")):
            return image_url

    if "image_path" in product and pd.notna(product["image_path"]):
        candidate = BASE_DIR / str(product["image_path"])

        if candidate.exists():
            return candidate

    product_slug = slugify(product_name)

    for extension in [".jpg", ".jpeg", ".png", ".webp", ".svg"]:
        candidate = PRODUCT_IMAGE_DIR / f"{product_slug}{extension}"

        if candidate.exists():
            return candidate

    category_slug = slugify(category)
    category_placeholder = PLACEHOLDER_IMAGE_DIR / f"{category_slug}.svg"

    if category_placeholder.exists():
        return category_placeholder

    return PLACEHOLDER_IMAGE_DIR / "default.svg"


def find_product_row(data, product_name):
    matched = data[data["product_name"] == product_name]

    if matched.empty:
        return None

    return matched.iloc[0]


def select_product(product_name):
    st.session_state["selected_product_name"] = product_name
    st.session_state["scroll_to_product_detail"] = True
    st.session_state["product_detail_scroll_nonce"] = (
        st.session_state.get("product_detail_scroll_nonce", 0) + 1
    )


def clear_selected_product():
    st.session_state.pop("selected_product_name", None)


def rerun_app():
    if hasattr(st, "rerun"):
        st.rerun()
    elif hasattr(st, "experimental_rerun"):
        st.experimental_rerun()


def get_product_description(product):
    if "description" in product and pd.notna(product["description"]):
        return str(product["description"])

    product_name = product["product_name"]
    category = product["category"]
    shop_name = product["shop_name"]
    price = format_price(product["price"])
    buyer_count = f"{int(product['buyer_count']):,}".replace(",", ".")
    rating = product["rating"]

    return (
        f"{product_name} là sản phẩm thuộc danh mục {category}, được bán bởi "
        f"{shop_name}. Sản phẩm có giá {price}, hiện ghi nhận "
        f"{buyer_count} lượt mua và điểm đánh giá {rating}/5. "
        f"Dựa trên các thông tin này, sản phẩm được đưa vào các mô hình K-Means "
        f"để phân tích nhóm giá, mức độ quan tâm của người mua và đặc điểm tổng hợp."
    )


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


def show_product_detail(product, price_df, segment_df, full_df):
    product_name = product["product_name"]
    scroll_nonce = st.session_state.get("product_detail_scroll_nonce", 0)
    anchor_id = f"product-detail-anchor-{scroll_nonce}"

    st.markdown(
        f'<div id="{anchor_id}" style="scroll-margin-top: 12px;"></div>',
        unsafe_allow_html=True
    )

    if st.session_state.pop("scroll_to_product_detail", False):
        components.html(
            f"""
            <script>
            const anchorId = "{anchor_id}";
            let tries = 0;

            function scrollToProductDetail() {{
                const parentWindow = window.parent;
                const doc = parentWindow.document;
                const anchor = doc.getElementById(anchorId);

                if (anchor) {{
                    const top = anchor.getBoundingClientRect().top + parentWindow.scrollY - 12;
                    parentWindow.scrollTo({{ top: Math.max(top, 0), behavior: "smooth" }});
                    return;
                }}

                tries += 1;
                if (tries < 8) {{
                    setTimeout(scrollToProductDetail, 150);
                }} else {{
                    parentWindow.scrollTo({{ top: 0, behavior: "smooth" }});
                }}
            }}

            setTimeout(scrollToProductDetail, 150);
            </script>
            """,
            height=0
        )

    st.divider()

    title_col, action_col = st.columns([5, 1])

    with title_col:
        st.subheader(product_name)

    with action_col:
        st.button(
            "Quay lại danh sách",
            on_click=clear_selected_product
        )

    image_col, info_col = st.columns([1, 2])

    with image_col:
        st.image(
            str(find_local_image(product)),
            use_container_width=True
        )

    with info_col:
        st.metric("Giá bán", format_price(product["price"]))
        st.write(f"**Shop:** {product['shop_name']}")
        st.write(f"**Danh mục:** {product['category']}")
        st.write(f"**Lượt mua:** {int(product['buyer_count']):,}".replace(",", "."))
        st.write(f"**Đánh giá:** {product['rating']}")

        st.write("### Mô tả sản phẩm")
        st.write(get_product_description(product))

        price_row = find_product_row(price_df, product_name)
        segment_row = find_product_row(segment_df, product_name)
        full_row = find_product_row(full_df, product_name)

        st.write("### Kết quả phân cụm")

        if price_row is not None and "price_group" in price_row:
            st.write(f"**Nhóm giá:** {display_label(price_row['price_group'])}")

        if segment_row is not None and "product_segment" in segment_row:
            st.write(f"**Nhóm sản phẩm:** {display_label(segment_row['product_segment'])}")

        if full_row is not None and "cluster_meaning" in full_row:
            st.write(f"**Cụm full feature:** {full_row['cluster_meaning']}")
        elif full_row is not None and "cluster" in full_row:
            st.write(f"**Cụm full feature:** Cụm {full_row['cluster']}")


def product_card(product):
    product_name = product["product_name"]

    st.image(
        str(find_local_image(product)),
        use_container_width=True
    )
    st.markdown(f"**{product_name}**")
    st.markdown(f"<span style='color:#ee4d2d;font-size:18px'>{format_price(product['price'])}</span>", unsafe_allow_html=True)
    st.caption(
        f"{product['category']} | ⭐ {product['rating']} | "
        f"{int(product['buyer_count']):,} lượt mua".replace(",", ".")
    )

    if st.button("Xem chi tiết", key=f"product_detail_{slugify(product_name)}"):
        select_product(product_name)
        rerun_app()


def get_related_products(data, selected_product):
    product_name = selected_product["product_name"]
    category = selected_product["category"]

    related = data[data["product_name"] != product_name].copy()
    same_category = related[related["category"] == category]
    other_category = related[related["category"] != category]

    same_category = same_category.sort_values(
        ["buyer_count", "rating"],
        ascending=[False, False]
    )
    other_category = other_category.sort_values(
        ["buyer_count", "rating"],
        ascending=[False, False]
    )

    return pd.concat([same_category, other_category])


def show_related_products(data, selected_product):
    related_products = get_related_products(data, selected_product)

    st.write("### Sản phẩm liên quan")
    st.caption(
        "Ưu tiên hiển thị sản phẩm cùng danh mục trước, sau đó đến các danh mục khác."
    )

    for start in range(0, min(len(related_products), 12), 4):
        columns = st.columns(4)
        rows = related_products.iloc[start:start + 4]

        for offset, (_, product) in enumerate(rows.iterrows()):
            with columns[offset]:
                product_card(product)


def product_gallery(data, price_df, segment_df, full_df):
    st.subheader("Danh sách sản phẩm")

    search = st.text_input(
        "Tìm kiếm trong danh sách sản phẩm",
        placeholder="Nhập tên sản phẩm, shop hoặc danh mục...",
        key="product_gallery_search"
    )

    filtered = data.copy()
    category_options = ["Tất cả"] + sorted(filtered["category"].dropna().unique())
    selected_category = st.selectbox(
        "Lọc danh mục sản phẩm",
        category_options,
        key="product_gallery_category"
    )

    if selected_category != "Tất cả":
        filtered = filtered[filtered["category"] == selected_category]

    if search:
        keyword = search.strip()
        filtered = filtered[
            filtered["product_name"].str.contains(keyword, case=False, na=False)
            | filtered["shop_name"].str.contains(keyword, case=False, na=False)
            | filtered["category"].str.contains(keyword, case=False, na=False)
        ]

    st.caption(f"Tìm thấy {len(filtered)} sản phẩm")

    selected_product_name = st.session_state.get("selected_product_name")

    if selected_product_name:
        selected_product = find_product_row(data, selected_product_name)

        if selected_product is not None:
            show_product_detail(selected_product, price_df, segment_df, full_df)
            show_related_products(data, selected_product)
            return

    for start in range(0, len(filtered), 4):
        columns = st.columns(4)
        rows = filtered.iloc[start:start + 4]

        for offset, (_, product) in enumerate(rows.iterrows()):
            with columns[offset]:
                product_card(product)


def main():
    st.title("Dashboard K-Means Sản Phẩm Shopee")
    st.write(
        "Dashboard hiển thị bảng dữ liệu, tìm kiếm sản phẩm và các biểu đồ "
        "phân cụm K-Means theo giá, lượt mua, đánh giá và đầy đủ đặc trưng."
    )

    if not DATA_PATH.exists():
        st.error(f"Không tìm thấy `{DATA_PATH.relative_to(BASE_DIR)}`.")
        return

    dataset = prepare_data(read_csv(DATA_PATH))
    price_result_df = load_result_if_exists(PRICE_RESULT_PATH)
    segment_result_df = load_result_if_exists(SEGMENT_RESULT_PATH)
    full_feature_result_df = load_result_if_exists(FULL_FEATURE_RESULT_PATH)

    total_products = len(dataset)
    average_price = dataset["price"].mean()
    total_buyers = dataset["buyer_count"].sum()
    average_rating = dataset["rating"].mean()

    metric_cols = st.columns(4)
    metric_cols[0].metric("Số sản phẩm", f"{total_products}")
    metric_cols[1].metric("Giá trung bình", format_price(average_price))
    metric_cols[2].metric("Tổng lượt mua", f"{int(total_buyers):,}".replace(",", "."))
    metric_cols[3].metric("Đánh giá trung bình", f"{average_rating:.2f}")

    dataset_tab, product_tab, price_tab, segment_tab, full_feature_tab = st.tabs(
        [
            "Dữ liệu & Tìm kiếm",
            "Sản phẩm",
            "K-Means Giá bán",
            "K-Means Hot/Ít mua/Đánh giá",
            "K-Means Đầy đủ đặc trưng"
        ]
    )

    with dataset_tab:
        searchable_table(dataset)

    with product_tab:
        product_gallery(
            dataset,
            price_result_df,
            segment_result_df,
            full_feature_result_df
        )

    with price_tab:
        st.subheader("Phân cụm theo giá bán")

        if PRICE_RESULT_PATH.exists():
            price_df = prepare_data(read_csv(PRICE_RESULT_PATH))

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
            segment_df = prepare_data(read_csv(SEGMENT_RESULT_PATH))

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
            full_df = prepare_data(read_csv(FULL_FEATURE_RESULT_PATH))

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
