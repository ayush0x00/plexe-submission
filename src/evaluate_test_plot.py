"""
Run experiments/delivery_date_nn.pt on test data and plot actual vs predicted vs estimated delivery time.
Run: source ~/base/bin/activate && python -m src.evaluate_test_plot
"""
from pathlib import Path
import os
import numpy as np
import pandas as pd
import torch
os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt

from src.delivery_dataset import load_and_build_features, FEATURE_COLS, prepare_X_y
from src.delivery_model import build_model

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_PATH = PROJECT_ROOT / "experiments" / "delivery_date_nn_2layer.pt"
OUT_DIR = PROJECT_ROOT / "experiments" / "graphs"
DATA_DIR = PROJECT_ROOT / "data"
SEED = 42


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main():
    device = get_device()
    print("Loading data and splitting 60/20/20...")
    df = load_and_build_features(DATA_DIR)
    rng = np.random.default_rng(SEED)
    n_total = len(df)
    indices = np.arange(n_total)
    rng.shuffle(indices)
    n_train = int(0.6 * n_total)
    n_val = int(0.2 * n_total)
    n_test = n_total - n_train - n_val
    train_idx = indices[:n_train]
    val_idx = indices[n_train : n_train + n_val]
    test_idx = indices[n_train + n_val :]

    train_df = df.iloc[train_idx].copy()
    val_df = df.iloc[val_idx].copy()
    test_df = df.iloc[test_idx].copy()

    # Same encoding as get_dataloaders
    train_df["category_encoded"] = pd.Categorical(train_df["category_mode"]).codes
    cat = pd.Categorical(train_df["category_mode"])
    test_df["category_encoded"] = pd.Categorical(test_df["category_mode"], categories=cat.categories).codes
    train_df["customer_state_code"] = pd.Categorical(train_df["customer_state"]).codes
    cat_c = pd.Categorical(train_df["customer_state"])
    test_df["customer_state_code"] = pd.Categorical(test_df["customer_state"], categories=cat_c.categories).codes
    train_df["seller_state_code"] = pd.Categorical(train_df["primary_seller_state"]).codes
    cat_s = pd.Categorical(train_df["primary_seller_state"])
    test_df["seller_state_code"] = pd.Categorical(test_df["primary_seller_state"], categories=cat_s.categories).codes

    X_train, y_train, scaler = prepare_X_y(train_df, target="delivery_days", fit_scaler=True)
    # Test: build X and get mask so we can index estimated_days
    X_test_full = test_df[[c for c in FEATURE_COLS if c in test_df.columns]].copy()
    X_test_full = X_test_full.fillna(X_test_full.median()).replace([np.inf, -np.inf], np.nan).fillna(X_test_full.median())
    mask = X_test_full.notna().all(axis=1)
    X_test = scaler.transform(X_test_full[mask].astype(np.float32))
    y_test = test_df.loc[mask, "delivery_days"].values
    estimated_days_test = test_df.loc[mask, "estimated_days"].values

    print(f"Test samples: {len(y_test)}")
    print("Loading model from", CHECKPOINT_PATH)
    ckpt = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    model = build_model(
        input_dim=ckpt["n_features"],
        hidden_dim=ckpt["hidden_dim"],
        num_hidden=ckpt["num_hidden"],
        dropout=0.0,
        device=device,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    X_t = torch.tensor(X_test, dtype=torch.float32, device=device)
    with torch.no_grad():
        pred_test = model(X_t).cpu().numpy()

    # If model was trained with clipped target, clip predictions to same range
    delivery_days_clip = ckpt.get("delivery_days_clip")
    if delivery_days_clip is not None:
        low, high = delivery_days_clip
        pred_test = np.clip(pred_test, low, high)
        print(f"Predictions clipped to {delivery_days_clip} (training used clipped target)")

    # Single line graph: x = sample index, y = days (three lines)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    n = len(y_test)
    x_axis = np.arange(n)
    plt.figure(figsize=(12, 5))
    plt.plot(x_axis, y_test, label="Actual delivery time (days)", color="C0", alpha=0.8)
    plt.plot(x_axis, pred_test, label="Predicted delivery time (model)", color="C1", alpha=0.8)
    plt.plot(x_axis, estimated_days_test, label="Estimated delivery time (data)", color="C2", alpha=0.8)
    plt.xlabel("Test sample index")
    plt.ylabel("Days")
    plt.title("Test set: actual vs predicted vs estimated delivery time")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path = OUT_DIR / "test_actual_vs_predicted_vs_estimated.png"
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"Plot saved: {out_path}")

    # Summary metrics
    mae_pred = np.mean(np.abs(pred_test - y_test))
    mae_est = np.mean(np.abs(estimated_days_test - y_test))
    print(f"MAE (model prediction vs actual): {mae_pred:.4f} days")
    print(f"MAE (data estimate vs actual):    {mae_est:.4f} days")


if __name__ == "__main__":
    main()
