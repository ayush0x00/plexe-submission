"""
PyTorch Dataset and DataLoader for delivery-date / late-delivery prediction.
Uses the same features as pca_delivery.py. 60:20:20 train:val:test split.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

# Default data path: project root / data
def _default_data_path():
    return Path(__file__).resolve().parent.parent / "data"


def load_and_build_features(data_dir=None):
    """Build order-level features and targets. Returns (df with feature cols + targets)."""
    data_dir = data_dir or _default_data_path()
    orders = pd.read_csv(data_dir / "olist_orders_dataset.csv", parse_dates=[
        "order_purchase_timestamp", "order_approved_at",
        "order_delivered_customer_date", "order_estimated_delivery_date",
    ])
    items = pd.read_csv(data_dir / "olist_order_items_dataset.csv")
    items["shipping_limit_date"] = pd.to_datetime(items["shipping_limit_date"])
    customers = pd.read_csv(data_dir / "olist_customers_dataset.csv")
    products = pd.read_csv(data_dir / "olist_products_dataset.csv")
    sellers = pd.read_csv(data_dir / "olist_sellers_dataset.csv")
    cat_trans = pd.read_csv(data_dir / "product_category_name_translation.csv")

    delivered = orders[orders["order_status"] == "delivered"].copy()
    delivered = delivered.dropna(subset=["order_delivered_customer_date", "order_approved_at", "order_estimated_delivery_date"])
    delivered["delivery_days"] = (delivered["order_delivered_customer_date"] - delivered["order_approved_at"]).dt.total_seconds() / 86400
    delivered["estimated_days"] = (delivered["order_estimated_delivery_date"] - delivered["order_approved_at"]).dt.total_seconds() / 86400
    delivered["delay_days"] = delivered["delivery_days"] - delivered["estimated_days"]
    delivered["is_late"] = (delivered["delay_days"] > 0).astype(int)

    order_agg = items.groupby("order_id").agg(
        total_price=("price", "sum"),
        total_freight=("freight_value", "sum"),
        n_items=("order_item_id", "count"),
        n_sellers=("seller_id", "nunique"),
    ).reset_index()
    primary_seller = items.groupby("order_id")["seller_id"].first().reset_index(name="primary_seller_id")
    order_agg = order_agg.merge(primary_seller, on="order_id")

    approval = delivered[["order_id", "order_approved_at"]]
    items = items.merge(approval, on="order_id", how="inner")
    items["shipping_slack_days"] = (items["shipping_limit_date"] - items["order_approved_at"]).dt.total_seconds() / 86400
    slack_agg = items.groupby("order_id")["shipping_slack_days"].min().reset_index(name="shipping_slack_days")
    order_agg = order_agg.merge(slack_agg, on="order_id")

    products_cat = products.merge(cat_trans, on="product_category_name", how="left")
    products_cat["product_category_name_english"] = products_cat["product_category_name_english"].fillna("unknown")
    items_p = items.merge(products_cat[["product_id", "product_weight_g", "product_category_name_english"]], on="product_id", how="left")
    weight_agg = items_p.groupby("order_id").agg(total_weight_g=("product_weight_g", "sum")).reset_index()
    cat_mode = items_p.groupby("order_id")["product_category_name_english"].agg(
        lambda x: x.mode().iloc[0] if len(x.mode()) else "unknown"
    ).reset_index(name="category_mode")
    order_agg = order_agg.merge(weight_agg, on="order_id").merge(cat_mode, on="order_id")

    delivered_with_items = delivered[["order_id", "delivery_days", "is_late"]].merge(items[["order_id", "seller_id"]], on="order_id")
    seller_hist = delivered_with_items.groupby("seller_id").agg(
        seller_mean_delivery_days=("delivery_days", "mean"),
        seller_late_rate=("is_late", "mean"),
    ).reset_index().rename(columns={"seller_id": "primary_seller_id"})
    order_agg = order_agg.merge(seller_hist, on="primary_seller_id", how="left")

    df = delivered[["order_id", "customer_id", "order_approved_at", "order_purchase_timestamp", "estimated_days", "delivery_days", "is_late"]].merge(order_agg, on="order_id")
    df = df.merge(customers[["customer_id", "customer_state"]], on="customer_id", how="left")
    df = df.merge(sellers.rename(columns={"seller_id": "primary_seller_id", "seller_state": "primary_seller_state"})[["primary_seller_id", "primary_seller_state"]], on="primary_seller_id", how="left")

    df["approval_delay_hours"] = (df["order_approved_at"] - df["order_purchase_timestamp"]).dt.total_seconds() / 3600
    df["purchase_weekday"] = df["order_purchase_timestamp"].dt.weekday
    df["purchase_hour"] = df["order_purchase_timestamp"].dt.hour
    df["purchase_month"] = df["order_purchase_timestamp"].dt.month
    df["same_state"] = (df["customer_state"] == df["primary_seller_state"]).astype(int)

    return df


FEATURE_COLS = [
    "estimated_days", "approval_delay_hours", "purchase_weekday", "purchase_hour", "purchase_month",
    "total_price", "total_freight", "n_items", "n_sellers", "shipping_slack_days", "total_weight_g",
    "seller_mean_delivery_days", "seller_late_rate", "same_state",
    "category_encoded", "customer_state_code", "seller_state_code",
]


def prepare_X_y(df, feature_cols=None, target="is_late", scaler=None, fit_scaler=False):
    """
    Build X (numeric) and y from df. Encodes categories and fills NaNs.
    Returns (X, y, scaler, feature_cols). If scaler is provided, use it to transform; else fit new one if fit_scaler=True.
    """
    feature_cols = feature_cols or FEATURE_COLS
    df = df.copy()
    if "category_encoded" not in df.columns and "category_mode" in df.columns:
        df["category_encoded"] = pd.Categorical(df["category_mode"]).codes
    if "customer_state_code" not in df.columns and "customer_state" in df.columns:
        df["customer_state_code"] = pd.Categorical(df["customer_state"]).codes
    if "seller_state_code" not in df.columns and "primary_seller_state" in df.columns:
        df["seller_state_code"] = pd.Categorical(df["primary_seller_state"]).codes

    X = df[[c for c in feature_cols if c in df.columns]].copy()
    X = X.fillna(X.median())
    X = X.replace([np.inf, -np.inf], np.nan).fillna(X.median())
    y = df[target].values

    mask = X.notna().all(axis=1)
    X = X[mask].astype(np.float32)
    y = y[mask]

    if fit_scaler:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    elif scaler is not None:
        X = scaler.transform(X)
    else:
        X = X.values

    return X, y, scaler


class DeliveryDataset(Dataset):
    """PyTorch Dataset for (X, y) tensors."""

    def __init__(self, X, y, dtype=torch.float32):
        self.X = torch.as_tensor(X, dtype=dtype)
        self.y = torch.as_tensor(y, dtype=torch.float32 if y.dtype.kind == "f" else torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def _iqr_bounds(series, multiplier=1.5, min_low=0.0):
    """IQR-based bounds: [Q1 - k*IQR, Q3 + k*IQR]. Returns (low, high)."""
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    low = max(min_low, float(q1 - multiplier * iqr))
    high = float(q3 + multiplier * iqr)
    return (low, high)


def _middle_percentile_bounds(series, middle_pct=75, min_low=0.0):
    """Keep middle middle_pct%, remove the rest from extremities. E.g. 75 -> [P12.5, P87.5]. Returns (low, high)."""
    tail = (100 - middle_pct) / 2
    low = max(min_low, float(series.quantile(tail / 100)))
    high = float(series.quantile(1 - tail / 100))
    return (low, high)


def get_dataloaders(
    data_dir=None,
    target="is_late",
    batch_size=64,
    seed=42,
    num_workers=0,
    delivery_days_clip=None,
    delivery_days_iqr=True,
    delivery_days_middle_pct=75,
):
    """
    Load data, 60:20:20 train:val:test split, scale using train only, return DataLoaders.
    target: "is_late" (binary) or "delivery_days" (regression).
    delivery_days_clip: optional (low, high) to clip delivery_days target. If None and delivery_days_iqr=True, uses IQR/middle-pct on train.
    delivery_days_iqr: if True and target=="delivery_days", compute clip from train when delivery_days_clip is None.
    delivery_days_middle_pct: when delivery_days_clip is None, keep middle this % and remove extremities (e.g. 75 -> [P12.5, P87.5]). Uses percentiles, not 1.5*IQR.
    Returns: train_loader, val_loader, test_loader, info_dict (with n_train, n_val, n_test, n_features, scaler, feature_cols, delivery_days_clip).
    """
    data_dir = data_dir or _default_data_path()
    rng = np.random.default_rng(seed)

    df = load_and_build_features(data_dir)
    n_total = len(df)
    indices = np.arange(n_total)
    rng.shuffle(indices)
    n_train = int(0.6 * n_total)
    n_val = int(0.2 * n_total)
    n_test = n_total - n_train - n_val
    train_idx, val_idx, test_idx = indices[:n_train], indices[n_train : n_train + n_val], indices[n_train + n_val :]

    train_df = df.iloc[train_idx].copy()
    val_df = df.iloc[val_idx].copy()
    test_df = df.iloc[test_idx].copy()

    # Clip delivery_days: use middle-pct (remove 75% of extremities = keep middle 75%) or explicit clip
    if target == "delivery_days":
        if delivery_days_clip is None and delivery_days_iqr:
            delivery_days_clip = _middle_percentile_bounds(
                train_df["delivery_days"], middle_pct=delivery_days_middle_pct, min_low=0.0
            )
            print(f"  delivery_days bounds (train, middle {delivery_days_middle_pct}%): {delivery_days_clip}")
        if delivery_days_clip is not None:
            low, high = delivery_days_clip
            for _df in (train_df, val_df, test_df):
                _df["delivery_days"] = _df["delivery_days"].clip(lower=low, upper=high)

    # Encode categories consistently (use train set's categorical mapping for val/test)
    for col, src in [("category_mode", "product_category_name_english"), ("customer_state", "customer_state"), ("primary_seller_state", "seller_state")]:
        if col == "category_mode":
            train_df["category_encoded"] = pd.Categorical(train_df["category_mode"]).codes
            cat = pd.Categorical(train_df["category_mode"])
            val_df["category_encoded"] = pd.Categorical(val_df["category_mode"], categories=cat.categories).codes
            test_df["category_encoded"] = pd.Categorical(test_df["category_mode"], categories=cat.categories).codes
        elif col == "customer_state":
            cat = pd.Categorical(train_df["customer_state"])
            train_df["customer_state_code"] = pd.Categorical(train_df["customer_state"]).codes
            val_df["customer_state_code"] = pd.Categorical(val_df["customer_state"], categories=cat.categories).codes
            test_df["customer_state_code"] = pd.Categorical(test_df["customer_state"], categories=cat.categories).codes
        else:
            cat = pd.Categorical(train_df["primary_seller_state"])
            train_df["seller_state_code"] = pd.Categorical(train_df["primary_seller_state"]).codes
            val_df["seller_state_code"] = pd.Categorical(val_df["primary_seller_state"], categories=cat.categories).codes
            test_df["seller_state_code"] = pd.Categorical(test_df["primary_seller_state"], categories=cat.categories).codes

    # Fit scaler on train only
    X_train, y_train, scaler = prepare_X_y(train_df, target=target, fit_scaler=True)
    X_val, y_val, _ = prepare_X_y(val_df, target=target, scaler=scaler)
    X_test, y_test, _ = prepare_X_y(test_df, target=target, scaler=scaler)
    n_features = X_train.shape[1]

    # Scaling check: log target range when regression
    if target == "delivery_days":
        print(f"  Target delivery_days: min={float(np.min(y_train)):.2f}, max={float(np.max(y_train)):.2f}, mean={float(np.mean(y_train)):.2f}")
        if delivery_days_clip:
            print(f"  (clipped to {delivery_days_clip} for training)")

    train_ds = DeliveryDataset(X_train, y_train)
    val_ds = DeliveryDataset(X_val, y_val)
    test_ds = DeliveryDataset(X_test, y_test)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    info = {
        "n_total": n_total,
        "n_train": len(y_train),
        "n_val": len(y_val),
        "n_test": len(y_test),
        "n_features": n_features,
        "feature_cols": FEATURE_COLS,
        "scaler": scaler,
        "target": target,
        "delivery_days_clip": delivery_days_clip,
    }
    return train_loader, val_loader, test_loader, info


if __name__ == "__main__":
    train_loader, val_loader, test_loader, info = get_dataloaders(target="is_late", batch_size=64, seed=42)
    print("Data points:")
    print(f"  Total:  {info['n_total']}")
    print(f"  Train:  {info['n_train']} (60%)")
    print(f"  Val:    {info['n_val']} (20%)")
    print(f"  Test:   {info['n_test']} (20%)")
    print(f"Features: {info['n_features']} ({info['feature_cols']})")
    print(f"Target: {info['target']}")
    x, y = next(iter(train_loader))
    print(f"Batch: X {x.shape}, y {y.shape}")
