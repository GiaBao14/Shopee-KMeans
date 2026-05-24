from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler


DATA_PATH = Path("data/data.csv")
OUTPUT_PATH = Path("output/kmeans_product_segment_result.csv")
CHART_PATH = Path("output/kmeans_product_segment_chart.png")
N_CLUSTERS = 3
LABEL_LIMIT_PER_SEGMENT = 5

HOT_SEGMENT = "Sản phẩm hot"
LOW_BUYER_SEGMENT = "Sản phẩm ít người mua"
HIGH_RATING_SEGMENT = "Sản phẩm đánh giá cao"


def get_labeled_products(data):
    labeled_groups = []

    for segment, group in data.groupby("product_segment"):
        if segment == HOT_SEGMENT:
            labeled_group = group.sort_values(
                ["buyer_count", "rating"],
                ascending=[False, False]
            ).head(LABEL_LIMIT_PER_SEGMENT)
        elif segment == LOW_BUYER_SEGMENT:
            labeled_group = group.sort_values(
                ["buyer_count", "rating"],
                ascending=[True, False]
            ).head(LABEL_LIMIT_PER_SEGMENT)
        else:
            labeled_group = group.sort_values(
                ["rating", "buyer_count"],
                ascending=[False, False]
            ).head(LABEL_LIMIT_PER_SEGMENT)

        labeled_groups.append(labeled_group)

    return pd.concat(labeled_groups).drop_duplicates("product_name")


def get_label_position(row, max_rating, index):
    if row["rating"] >= max_rating - 0.01:
        offsets = [
            (0, -28),
            (28, -38),
            (-28, -48),
            (45, -58),
            (-45, -68)
        ]
        return offsets[index % len(offsets)], "top"

    offsets = [
        (0, 14),
        (24, 24),
        (-24, 34),
        (42, 44),
        (-42, 54)
    ]
    return offsets[index % len(offsets)], "bottom"


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
    "buyer_count",
    "rating"
]

for column in numeric_columns:
    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )

invalid_rows = df[df[numeric_columns].isnull().any(axis=1)]

if not invalid_rows.empty:
    raise ValueError(
        "Có sản phẩm có buyer_count/rating không hợp lệ:\n"
        + invalid_rows[
            [
                "product_name",
                "buyer_count",
                "rating"
            ]
        ].to_string(index=False)
    )

# =========================
# K-MEANS THEO LUOT MUA VA DANH GIA
# =========================

features = [
    "buyer_count",
    "rating"
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

low_buyer_cluster = cluster_centers.sort_values("buyer_count").iloc[0]["cluster"]

remaining_clusters = cluster_centers[
    cluster_centers["cluster"] != low_buyer_cluster
]

hot_cluster = remaining_clusters.sort_values(
    "buyer_count",
    ascending=False
).iloc[0]["cluster"]

high_rating_cluster = remaining_clusters[
    remaining_clusters["cluster"] != hot_cluster
].iloc[0]["cluster"]

segment_map = {
    int(hot_cluster): HOT_SEGMENT,
    int(low_buyer_cluster): LOW_BUYER_SEGMENT,
    int(high_rating_cluster): HIGH_RATING_SEGMENT
}

df["product_segment"] = df["cluster"].map(segment_map)

print("\n===== TÂM CỤM K-MEANS =====")
print(
    cluster_centers
    .assign(segment=cluster_centers["cluster"].map(segment_map))
    .sort_values("buyer_count")
)

print("\n===== KẾT QUẢ PHÂN CỤM SẢN PHẨM =====")
print(df[[
    "product_name",
    "buyer_count",
    "rating",
    "product_segment"
]])

print("\n===== SỐ LƯỢNG SẢN PHẨM THEO NHÓM =====")
print(df["product_segment"].value_counts())

# =========================
# VE BIEU DO
# =========================

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams["font.family"] = "DejaVu Sans"

fig, ax = plt.subplots(figsize=(13, 7))

segment_colors = {
    HOT_SEGMENT: "#e15759",
    LOW_BUYER_SEGMENT: "#4e79a7",
    HIGH_RATING_SEGMENT: "#59a14f"
}

segment_order = [
    HOT_SEGMENT,
    LOW_BUYER_SEGMENT,
    HIGH_RATING_SEGMENT
]

for segment in segment_order:
    group = df[df["product_segment"] == segment]

    ax.scatter(
        group["buyer_count"],
        group["rating"],
        s=90,
        alpha=0.88,
        label=segment,
        color=segment_colors[segment],
        edgecolor="white",
        linewidth=0.8,
        zorder=3
    )

representative_rows = get_labeled_products(df)
max_rating = df["rating"].max()

for _, group in representative_rows.groupby("product_segment"):
    for index, (_, row) in enumerate(group.iterrows()):
        offset, vertical_align = get_label_position(row, max_rating, index)

        ax.annotate(
            row["product_name"],
            (row["buyer_count"], row["rating"]),
            xytext=offset,
            textcoords="offset points",
            ha="center",
            va=vertical_align,
            fontsize=8,
            color="#243447",
            bbox={
                "boxstyle": "round,pad=0.25",
                "facecolor": "white",
                "edgecolor": "#d0d7de",
                "alpha": 0.92
            },
            arrowprops={
                "arrowstyle": "-",
                "color": "#8c96a3",
                "linewidth": 0.6,
                "shrinkA": 0,
                "shrinkB": 5
            },
            zorder=4
        )

ax.set_title(
    "K-Means phân cụm sản phẩm theo lượt mua và đánh giá",
    fontsize=16,
    fontweight="bold",
    pad=18
)
ax.set_xlabel("Lượt mua", fontsize=11)
ax.set_ylabel("Đánh giá", fontsize=11)
ax.set_ylim(
    df["rating"].min() - 0.04,
    df["rating"].max() + 0.08
)

fig.text(
    0.76,
    0.58,
    "Ý nghĩa biểu đồ:\n"
    "- Trục X: lượt mua sản phẩm\n"
    "- Trục Y: điểm đánh giá\n"
    "- Màu đỏ: sản phẩm hot, lượt mua cao\n"
    "- Màu xanh dương: ít người mua\n"
    "- Màu xanh lá: đánh giá cao",
    ha="left",
    va="top",
    fontsize=9,
    color="#243447",
    bbox={
        "boxstyle": "round,pad=0.45",
        "facecolor": "white",
        "edgecolor": "#d0d7de",
        "alpha": 0.95
    }
)

ax.grid(axis="both", linestyle="--", linewidth=0.7, alpha=0.4)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend(
    title="Nhóm sản phẩm",
    frameon=False,
    loc="upper left",
    bbox_to_anchor=(1.02, 1)
)
ax.margins(x=0.08)

fig.subplots_adjust(right=0.72, top=0.86)

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
