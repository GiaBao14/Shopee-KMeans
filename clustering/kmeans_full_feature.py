from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler


DATA_PATH = Path("data/data.csv")
OUTPUT_PATH = Path("output/kmeans_full_feature_result.csv")
CHART_PATH = Path("output/kmeans_full_feature_chart.png")
N_CLUSTERS = 5
LABEL_LIMIT_PER_CLUSTER = 3


def format_million(value, _position):
    return f"{value / 1_000_000:.0f}M"


def get_labeled_products(data):
    labeled_groups = []

    for _, group in data.groupby("cluster"):
        labeled_group = group.sort_values(
            ["price", "buyer_count", "rating"],
            ascending=[False, False, False]
        ).head(LABEL_LIMIT_PER_CLUSTER)

        labeled_groups.append(labeled_group)

    return pd.concat(labeled_groups).drop_duplicates("product_name")


def get_cluster_meanings(cluster_centers):
    meanings = {}

    price_low = cluster_centers["price"].quantile(0.25)
    price_high = cluster_centers["price"].quantile(0.75)
    buyer_low = cluster_centers["buyer_count"].quantile(0.25)
    buyer_high = cluster_centers["buyer_count"].quantile(0.75)
    rating_low = cluster_centers["rating"].quantile(0.25)
    rating_high = cluster_centers["rating"].quantile(0.75)

    for _, row in cluster_centers.iterrows():
        descriptions = []

        if row["price"] >= price_high:
            descriptions.append("giá cao")
        elif row["price"] <= price_low:
            descriptions.append("giá thấp")

        if row["buyer_count"] >= buyer_high:
            descriptions.append("bán chạy")
        elif row["buyer_count"] <= buyer_low:
            descriptions.append("ít người mua")

        if row["rating"] >= rating_high:
            descriptions.append("đánh giá cao")
        elif row["rating"] <= rating_low:
            descriptions.append("đánh giá thấp")

        if not descriptions:
            descriptions.append("nhóm trung bình")

        cluster_id = int(row["cluster"])
        meanings[cluster_id] = f"C{cluster_id}: " + ", ".join(descriptions)

    return meanings


# =========================
# DOC DATA
# =========================

df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")

print("===== DATASET =====")
print(df.head())

# =========================
# XU LY DU LIEU
# =========================

df = df.drop_duplicates()

numeric_columns = [
    "price",
    "buyer_count",
    "rating"
]

for column in numeric_columns:
    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )

df["category"] = df["category"].fillna("Unknown")

invalid_rows = df[df[numeric_columns].isnull().any(axis=1)]

if not invalid_rows.empty:
    raise ValueError(
        "Có sản phẩm có price/buyer_count/rating không hợp lệ:\n"
        + invalid_rows[
            [
                "product_name",
                "price",
                "category",
                "buyer_count",
                "rating"
            ]
        ].to_string(index=False)
    )

# =========================
# ENCODE CATEGORY
# =========================

encoder = LabelEncoder()

df["category_encoded"] = encoder.fit_transform(
    df["category"]
)

print("\n===== CATEGORY ENCODE =====")
print(
    df[[
        "category",
        "category_encoded"
    ]]
    .drop_duplicates()
    .sort_values("category_encoded")
)

# =========================
# K-MEANS FULL FEATURE
# =========================

features = [
    "price",
    "buyer_count",
    "rating",
    "category_encoded"
]

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(df[features])

kmeans = KMeans(
    n_clusters=N_CLUSTERS,
    random_state=42,
    n_init=10
)

df["cluster"] = kmeans.fit_predict(X_scaled)

cluster_centers = pd.DataFrame(
    scaler.inverse_transform(kmeans.cluster_centers_),
    columns=features
)
cluster_centers["cluster"] = cluster_centers.index

cluster_meanings = get_cluster_meanings(cluster_centers)
df["cluster_meaning"] = df["cluster"].map(cluster_meanings)

print("\n===== TÂM CỤM K-MEANS FULL FEATURE =====")
print(
    cluster_centers[[
        "cluster",
        "price",
        "buyer_count",
        "rating",
        "category_encoded"
    ]].sort_values("cluster")
)

print("\n===== KẾT QUẢ PHÂN CỤM FULL FEATURE =====")
print(df[[
    "product_name",
    "category",
    "category_encoded",
    "price",
    "buyer_count",
    "rating",
    "cluster",
    "cluster_meaning"
]])

print("\n===== SỐ LƯỢNG SẢN PHẨM THEO CLUSTER =====")
print(df["cluster"].value_counts().sort_index())

print("\n===== THỐNG KÊ THEO CLUSTER =====")
print(
    df.groupby("cluster")[[
        "price",
        "buyer_count",
        "rating",
        "category_encoded"
    ]]
    .agg(["count", "min", "mean", "max"])
)

# =========================
# VE BIEU DO
# =========================

plt.style.use("seaborn-v0_8-whitegrid")

fig, ax = plt.subplots(figsize=(12, 7))

colors = [
    "#4e79a7",
    "#f28e2b",
    "#59a14f",
    "#e15759",
    "#b07aa1"
]

for cluster_id, group in df.groupby("cluster"):
    ax.scatter(
        group["buyer_count"],
        group["price"],
        s=group["rating"] * 25,
        alpha=0.82,
        label=cluster_meanings[int(cluster_id)],
        color=colors[int(cluster_id) % len(colors)],
        edgecolor="white",
        linewidth=0.8
    )

labeled_products = get_labeled_products(df)

label_offsets = [
    (0, 14),
    (22, 24),
    (-22, 34)
]

for _, group in labeled_products.groupby("cluster"):
    for index, (_, row) in enumerate(group.iterrows()):
        ax.annotate(
            row["product_name"],
            (row["buyer_count"], row["price"]),
            xytext=label_offsets[index % len(label_offsets)],
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
            color="#243447",
            bbox={
                "boxstyle": "round,pad=0.25",
                "facecolor": "white",
                "edgecolor": "#d0d7de",
                "alpha": 0.9
            },
            arrowprops={
                "arrowstyle": "-",
                "color": "#8c96a3",
                "linewidth": 0.6,
                "shrinkA": 0,
                "shrinkB": 5
            }
        )

ax.set_title(
    "K-Means full feature phân cụm sản phẩm",
    fontsize=15,
    fontweight="bold",
    pad=16
)
ax.set_xlabel("Lượt mua", fontsize=11)
ax.set_ylabel("Giá bán (VND)", fontsize=11)
ax.yaxis.set_major_formatter(format_million)
ax.text(
    0.02,
    0.98,
    "Ý nghĩa biểu đồ:\n"
    "- Trục X: lượt mua\n"
    "- Trục Y: giá ban\n"
    "- Chấm lớn hơn: rating cao hơn\n"
    "- Mau: cụm K-Means từ price, buyer_count, rating, category_encoded\n"
    "- category_encoded có tham gia phân cụm nhưng không vẽ thành trục riêng",
    transform=ax.transAxes,
    ha="left",
    va="top",
    fontsize=9,
    color="#243447",
    bbox={
        "boxstyle": "round,pad=0.45",
        "facecolor": "white",
        "edgecolor": "#d0d7de",
        "alpha": 0.92
    }
)
ax.grid(axis="both", linestyle="--", linewidth=0.7, alpha=0.4)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend(title="Y nghia cum", frameon=False, loc="upper right")
ax.margins(x=0.08, y=0.12)

fig.tight_layout()

# =========================
# LUU KET QUA
# =========================

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

df.to_csv(
    OUTPUT_PATH,
    index=False,
    encoding="utf-8-sig"
)

fig.savefig(
    CHART_PATH,
    dpi=200,
    bbox_inches="tight"
)

plt.show()

print(f"\nSaved cluster result: {OUTPUT_PATH}")
print(f"Saved chart: {CHART_PATH}")
