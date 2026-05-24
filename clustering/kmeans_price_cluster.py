from pathlib import Path
from textwrap import fill

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler


DATA_PATH = Path("data/data.csv")
OUTPUT_PATH = Path("output/kmeans_price_result.csv")
CHART_PATH = Path("output/top_6_highest_price_products.png")
N_CLUSTERS = 3


def format_price(value):
    return f"{int(value):,} VND".replace(",", ".")


def format_million(value, _position):
    return f"{value / 1_000_000:.0f}M"


def wrap_product_name(value):
    return fill(value, width=18)


# =========================
# DOC DATA
# =========================

df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")

print("===== DATASET =====")
print(df.head())

# =========================
# XU LY DU LIEU GIA
# =========================

df = df.drop_duplicates()

df["price"] = pd.to_numeric(
    df["price"],
    errors="coerce"
)

invalid_price_rows = df[df["price"].isnull()]

if not invalid_price_rows.empty:
    raise ValueError(
        "Co san pham co price khong hop le:\n"
        + invalid_price_rows[
            [
                "product_name",
                "price",
                "category"
            ]
        ].to_string(index=False)
    )

# =========================
# K-MEANS THEO GIA BAN
# =========================

scaler = MinMaxScaler()
price_scaled = scaler.fit_transform(df[["price"]])

kmeans = KMeans(
    n_clusters=N_CLUSTERS,
    random_state=42,
    n_init=10
)

df["price_cluster"] = kmeans.fit_predict(price_scaled)

cluster_centers = scaler.inverse_transform(
    kmeans.cluster_centers_
).flatten()

cluster_order = (
    pd.Series(cluster_centers)
    .sort_values()
    .index
    .tolist()
)

cluster_names = [
    "Gia thap",
    "Gia trung binh",
    "Gia cao"
]

cluster_name_map = {
    cluster_id: cluster_names[index]
    for index, cluster_id in enumerate(cluster_order)
}

df["price_group"] = df["price_cluster"].map(cluster_name_map)

print("\n===== KET QUA PHAN CUM GIA BAN =====")
print(df[[
    "product_name",
    "price",
    "price_cluster",
    "price_group"
]])

print("\n===== THONG KE THEO CUM GIA =====")
print(
    df.groupby("price_group")["price"]
    .agg(["count", "min", "mean", "max"])
    .sort_values("mean")
)

# =========================
# TOP 6 SAN PHAM GIA CAO NHAT
# =========================

top_6 = (
    df.sort_values("price", ascending=False)
    .head(6)
    .reset_index(drop=True)
)

print("\n===== TOP 6 SAN PHAM GIA CAO NHAT =====")
print(top_6[[
    "product_name",
    "price",
    "price_group"
]])

plt.style.use("seaborn-v0_8-whitegrid")

fig, ax = plt.subplots(figsize=(13, 7))

group_colors = {
    "Gia thap": "#59a14f",
    "Gia trung binh": "#f28e2b",
    "Gia cao": "#4e79a7"
}

bar_colors = top_6["price_group"].map(group_colors).fillna("#4e79a7")

bars = ax.bar(
    range(len(top_6)),
    top_6["price"],
    color=bar_colors,
    width=0.62,
    edgecolor="#243447",
    linewidth=0.8
)

ax.set_title(
    "Top 6 san pham co gia ban cao nhat",
    fontsize=18,
    fontweight="bold",
    pad=18
)
ax.set_xlabel("")
ax.set_ylabel("Gia ban (VND)", fontsize=11)
ax.set_xticks(
    range(len(top_6)),
    [wrap_product_name(name) for name in top_6["product_name"]],
    fontsize=9
)
ax.yaxis.set_major_formatter(format_million)
ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.45)
ax.grid(axis="x", visible=False)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_alpha(0.35)
ax.spines["bottom"].set_alpha(0.35)

max_price = top_6["price"].max()

for bar, product_name, price in zip(
    bars,
    top_6["product_name"],
    top_6["price"]
):
    height = bar.get_height()

    ax.text(
        bar.get_x() + bar.get_width() / 2,
        height + max_price * 0.035,
        format_price(price),
        ha="center",
        va="bottom",
        fontsize=9,
        fontweight="bold",
        color="#243447"
    )

    ax.text(
        bar.get_x() + bar.get_width() / 2,
        height * 0.52,
        wrap_product_name(product_name),
        ha="center",
        va="center",
        fontsize=8.5,
        color="white",
        fontweight="bold",
        linespacing=1.1
    )

legend_handles = [
    plt.Rectangle((0, 0), 1, 1, color=color)
    for color in group_colors.values()
]

ax.legend(
    legend_handles,
    group_colors.keys(),
    title="Cum gia",
    frameon=False,
    loc="upper right"
)

ax.set_ylim(0, max_price * 1.22)
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
