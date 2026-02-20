"""
PCA on delivery-date features to find which inputs matter most for predicting late delivery.
Run: source ~/base/bin/activate && python pca_delivery.py
"""
import os
BASE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("MPLCONFIGDIR", os.path.join(BASE, ".mplcache"))
os.environ.setdefault("MPLBACKEND", "Agg")
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

DATA = Path(BASE) / "data"
OUT = Path(BASE) / "experiments" / "graphs"
OUT.mkdir(parents=True, exist_ok=True)


def load_and_build_features():
    orders = pd.read_csv(DATA / "olist_orders_dataset.csv", parse_dates=[
        "order_purchase_timestamp", "order_approved_at",
        "order_delivered_customer_date", "order_estimated_delivery_date",
    ])
    items = pd.read_csv(DATA / "olist_order_items_dataset.csv")
    items["shipping_limit_date"] = pd.to_datetime(items["shipping_limit_date"])
    customers = pd.read_csv(DATA / "olist_customers_dataset.csv")
    products = pd.read_csv(DATA / "olist_products_dataset.csv")
    sellers = pd.read_csv(DATA / "olist_sellers_dataset.csv")
    cat_trans = pd.read_csv(DATA / "product_category_name_translation.csv")

    # Delivered only, with target
    delivered = orders[orders["order_status"] == "delivered"].copy()
    delivered = delivered.dropna(subset=["order_delivered_customer_date", "order_approved_at", "order_estimated_delivery_date"])
    delivered["delivery_days"] = (delivered["order_delivered_customer_date"] - delivered["order_approved_at"]).dt.total_seconds() / 86400
    delivered["estimated_days"] = (delivered["order_estimated_delivery_date"] - delivered["order_approved_at"]).dt.total_seconds() / 86400
    delivered["delay_days"] = delivered["delivery_days"] - delivered["estimated_days"]
    delivered["is_late"] = (delivered["delay_days"] > 0).astype(int)

    # Order-level aggregates from items
    order_agg = items.groupby("order_id").agg(
        total_price=("price", "sum"),
        total_freight=("freight_value", "sum"),
        n_items=("order_item_id", "count"),
        n_sellers=("seller_id", "nunique"),
    ).reset_index()
    # Primary seller (first by order)
    primary_seller = items.groupby("order_id")["seller_id"].first().reset_index(name="primary_seller_id")
    order_agg = order_agg.merge(primary_seller, on="order_id")
    # Shipping slack: days from approval to shipping_limit (min per order)
    approval = delivered[["order_id", "order_approved_at"]]
    items = items.merge(approval, on="order_id", how="inner")
    items["shipping_slack_days"] = (items["shipping_limit_date"] - items["order_approved_at"]).dt.total_seconds() / 86400
    slack_agg = items.groupby("order_id")["shipping_slack_days"].min().reset_index(name="shipping_slack_days")
    order_agg = order_agg.merge(slack_agg, on="order_id")

    # Product: weight and category (mode per order)
    products_cat = products.merge(cat_trans, on="product_category_name", how="left")
    products_cat["product_category_name_english"] = products_cat["product_category_name_english"].fillna("unknown")
    items_p = items.merge(products_cat[["product_id", "product_weight_g", "product_category_name_english"]], on="product_id", how="left")
    weight_agg = items_p.groupby("order_id").agg(
        total_weight_g=("product_weight_g", "sum"),
    ).reset_index()
    # Mode category per order
    cat_mode = items_p.groupby("order_id")["product_category_name_english"].agg(
        lambda x: x.mode().iloc[0] if len(x.mode()) else "unknown"
    ).reset_index(name="category_mode")
    order_agg = order_agg.merge(weight_agg, on="order_id").merge(cat_mode, on="order_id")

    # Seller historical stats (from all delivered orders - slight lookahead for PCA exploration)
    delivered_with_items = delivered[["order_id", "delivery_days", "is_late"]].merge(
        items[["order_id", "seller_id"]], on="order_id"
    )
    seller_hist = delivered_with_items.groupby("seller_id").agg(
        seller_mean_delivery_days=("delivery_days", "mean"),
        seller_late_rate=("is_late", "mean"),
    ).reset_index()
    seller_hist = seller_hist.rename(columns={"seller_id": "primary_seller_id"})
    order_agg = order_agg.merge(seller_hist, on="primary_seller_id", how="left")

    # Merge all into delivered
    df = delivered[["order_id", "customer_id", "order_purchase_timestamp", "order_approved_at", "estimated_days", "delivery_days", "is_late"]].merge(order_agg, on="order_id")
    df = df.merge(customers[["customer_id", "customer_state"]], on="customer_id", how="left")
    df = df.merge(sellers.rename(columns={"seller_id": "primary_seller_id", "seller_state": "primary_seller_state"})[["primary_seller_id", "primary_seller_state"]], on="primary_seller_id", how="left")

    # Derived: approval delay (hours), time features, same_state
    df["approval_delay_hours"] = (df["order_approved_at"] - df["order_purchase_timestamp"]).dt.total_seconds() / 3600
    df["purchase_weekday"] = df["order_purchase_timestamp"].dt.weekday
    df["purchase_hour"] = df["order_purchase_timestamp"].dt.hour
    df["purchase_month"] = df["order_purchase_timestamp"].dt.month
    df["same_state"] = (df["customer_state"] == df["primary_seller_state"]).astype(int)

    return df


def run_pca(df):
    # Numeric features only (no order_id, customer_id, timestamps, target)
    feature_cols = [
        "estimated_days", "approval_delay_hours", "purchase_weekday", "purchase_hour", "purchase_month",
        "total_price", "total_freight", "n_items", "n_sellers", "shipping_slack_days", "total_weight_g",
        "seller_mean_delivery_days", "seller_late_rate", "same_state",
    ]
    # Category: label encode for PCA
    if "category_mode" in df.columns:
        df["category_encoded"] = pd.Categorical(df["category_mode"]).codes
        feature_cols.append("category_encoded")
    # Customer state and seller state (label encoded)
    df["customer_state_code"] = pd.Categorical(df["customer_state"]).codes
    df["seller_state_code"] = pd.Categorical(df["primary_seller_state"]).codes
    feature_cols.extend(["customer_state_code", "seller_state_code"])

    X = df[feature_cols].copy()
    X = X.fillna(X.median())
    y = df["is_late"]

    # Drop any remaining inf/nan
    X = X.replace([np.inf, -np.inf], np.nan).fillna(X.median())
    mask = X.notna().all(axis=1)
    X = X[mask]
    y = y[mask]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    n_components = min(15, X.shape[1], X.shape[0] - 1)
    pca = PCA(n_components=n_components, random_state=42)
    pca.fit(X_scaled)
    components = pca.transform(X_scaled)

    # Correlation of each PC with is_late
    pc_names = [f"PC{i+1}" for i in range(components.shape[1])]
    corrs = [np.corrcoef(components[:, i], y)[0, 1] for i in range(components.shape[1])]

    loadings = pd.DataFrame(
        pca.components_.T,
        index=feature_cols,
        columns=pc_names,
    )
    return {
        "X": X,
        "y": y,
        "feature_cols": feature_cols,
        "scaler": scaler,
        "pca": pca,
        "components": components,
        "loadings": loadings,
        "pc_corr_with_late": dict(zip(pc_names, corrs)),
        "explained_variance_ratio": pca.explained_variance_ratio_,
    }


def main():
    print("Loading and building features...")
    df = load_and_build_features()
    print(f"Orders (delivered, with features): {len(df)}")
    print(f"Late rate: {df['is_late'].mean():.2%}")

    print("\nRunning PCA...")
    res = run_pca(df)

    print("\n" + "=" * 60)
    print("EXPLAINED VARIANCE (scree)")
    print("=" * 60)
    for i, r in enumerate(res["explained_variance_ratio"]):
        print(f"  PC{i+1}: {r:.3f}  (cumul: {res['explained_variance_ratio'][:i+1].sum():.3f})")
    print(f"  Total: {res['explained_variance_ratio'].sum():.3f}")

    print("\n" + "=" * 60)
    print("CORRELATION OF EACH PC WITH is_late (which components predict late?)")
    print("=" * 60)
    for pc, c in sorted(res["pc_corr_with_late"].items(), key=lambda x: -abs(x[1])):
        print(f"  {pc}: r = {c:+.3f}")

    # Which original features drive the PCs that correlate most with is_late?
    top_pcs_by_corr = sorted(res["pc_corr_with_late"].items(), key=lambda x: -abs(x[1]))[:3]
    print("\n" + "=" * 60)
    print("LOADINGS: Which original features drive the PCs that predict late?")
    print("(High |loading| = feature contributes a lot to that PC)")
    print("=" * 60)
    for pc, _ in top_pcs_by_corr:
        load = res["loadings"][pc].reindex(res["loadings"][pc].abs().sort_values(ascending=False).index)
        print(f"\n  {pc} (corr with is_late = {res['pc_corr_with_late'][pc]:+.3f}):")
        for feat in load.index[:10]:
            print(f"    {feat}: {load[feat]:+.3f}")

    # Importance summary: for PCs that correlate with is_late, sum absolute loadings per feature (weighted by |corr|)
    importance = pd.Series(0.0, index=res["feature_cols"])
    for pc, c in res["pc_corr_with_late"].items():
        importance += res["loadings"][pc].abs() * abs(c)
    importance = importance.sort_values(ascending=False)
    print("\n" + "=" * 60)
    print("FEATURE IMPORTANCE FOR PREDICTING LATE (PCA-based)")
    print("(Sum of |loading| * |corr(PC, is_late)| across PCs)")
    print("=" * 60)
    for feat in importance.index:
        print(f"  {feat}: {importance[feat]:.3f}")

    # Plots
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(10, 9))

    # Scree
    ax = axes[0]
    n_pc = len(res["explained_variance_ratio"])
    ax.bar(range(1, n_pc + 1), res["explained_variance_ratio"], alpha=0.8, label="Individual")
    ax.plot(range(1, n_pc + 1), np.cumsum(res["explained_variance_ratio"]), "o-", color="C1", label="Cumulative")
    ax.set_xlabel("Principal component")
    ax.set_ylabel("Explained variance ratio")
    ax.set_title("PCA: Scree plot (delivery-date features)")
    ax.legend()
    ax.set_xticks(range(1, n_pc + 1))

    # Loadings for top 2 PCs that correlate most with is_late
    ax = axes[1]
    top2 = [x[0] for x in top_pcs_by_corr[:2]]
    load_top = res["loadings"][top2]
    load_top = load_top.reindex(load_top.abs().max(axis=1).sort_values(ascending=False).index)
    load_top = load_top.head(14)
    load_top.plot(kind="barh", ax=ax, width=0.8)
    ax.set_title(f"Loadings on PCs most correlated with is_late: {top2[0]}, {top2[1]}")
    ax.set_xlabel("Loading")
    ax.legend(title="PC")
    plt.tight_layout()
    plt.savefig(OUT / "pca_delivery_scree_loadings.png", dpi=120)
    plt.close()

    # Bar: feature importance (PCA-based)
    fig, ax = plt.subplots(figsize=(9, 6))
    importance.sort_values().plot(kind="barh", ax=ax, color="steelblue", alpha=0.8)
    ax.set_title("Feature importance for predicting late delivery (PCA-based)")
    ax.set_xlabel("Importance (|loading| × |corr(PC, is_late)| summed)")
    plt.tight_layout()
    plt.savefig(OUT / "pca_delivery_importance.png", dpi=120)
    plt.close()

    print(f"\nPlots saved: {OUT / 'pca_delivery_scree_loadings.png'}, {OUT / 'pca_delivery_importance.png'}")


if __name__ == "__main__":
    main()
