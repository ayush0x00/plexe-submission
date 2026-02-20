"""
Train DeliveryDateNN for delivery_days regression. 4 hidden layers, 10 neurons each.
Run: source ~/base/bin/activate && python -m src.train
"""
from pathlib import Path
import torch
import torch.nn as nn

from src.delivery_dataset import get_dataloaders
from src.delivery_model import build_model

import os
os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "experiments" / "graphs"
CHECKPOINT_DIR = PROJECT_ROOT / "experiments"
def _get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


DEVICE = _get_device()


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    n = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device).float()
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        n += x.size(0)
    return total_loss / n if n else 0.0


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    n = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device).float()
            out = model(x)
            loss = criterion(out, y)
            total_loss += loss.item() * x.size(0)
            n += x.size(0)
    return total_loss / n if n else 0.0


def main(
    data_dir=None,
    batch_size=64,
    epochs=150,
    lr=1e-3,
    hidden_dim=10,
    num_hidden=4,
    seed=42,
    save_dir=None,
    run_name="4layer",
):
    torch.manual_seed(seed)
    print(f"Device: {DEVICE}")
    save_dir = save_dir or CHECKPOINT_DIR
    save_dir.mkdir(parents=True, exist_ok=True)
    ckpt_name = f"delivery_date_nn_{run_name}.pt"
    plot_name = f"train_val_loss_{run_name}.png"

    # Clip delivery_days by IQR: keep middle 75%, remove 25% from extremities (train percentiles)
    train_loader, val_loader, test_loader, info = get_dataloaders(
        data_dir=data_dir,
        target="delivery_days",
        batch_size=batch_size,
        seed=seed,
        delivery_days_clip=None,
        delivery_days_iqr=True,
        delivery_days_middle_pct=75,
    )
    n_features = info["n_features"]

    model = build_model(
        input_dim=n_features,
        hidden_dim=hidden_dim,
        num_hidden=num_hidden,
        dropout=0.0,
        device=DEVICE,
    )
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_losses, val_losses = [], []
    best_val = float("inf")
    val_above_train_streak = 0
    early_stop_patience = 3  # stop when val_loss > train_loss for this many consecutive epochs

    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, DEVICE)
        val_loss = evaluate(model, val_loader, criterion, DEVICE)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "scaler": info["scaler"],
                    "n_features": n_features,
                    "feature_cols": info["feature_cols"],
                    "hidden_dim": hidden_dim,
                    "num_hidden": num_hidden,
                    "delivery_days_clip": info.get("delivery_days_clip"),
                },
                save_dir / ckpt_name,
            )

        if val_loss > train_loss:
            val_above_train_streak += 1
            if val_above_train_streak >= early_stop_patience:
                print(f"Epoch {epoch:3d}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  [early stop: val > train for {early_stop_patience} epochs]")
                break
        else:
            val_above_train_streak = 0

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

    n_epochs_run = len(train_losses)
    # Train vs val loss plot
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, n_epochs_run + 1), train_losses, label="Train loss", color="C0")
    plt.plot(range(1, n_epochs_run + 1), val_losses, label="Val loss", color="C1")
    plt.xlabel("Epoch")
    plt.ylabel("MSE loss")
    plt.title(f"Delivery date NN ({run_name}): train vs val loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / plot_name, dpi=120)
    plt.close()
    print(f"Plot saved: {OUT_DIR / plot_name}")

    # Load best and report test
    ckpt = torch.load(save_dir / ckpt_name, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    test_loss = evaluate(model, test_loader, criterion, DEVICE)
    test_mae = evaluate(model, test_loader, nn.L1Loss(), DEVICE)
    print(f"\nBest model on test: MSE={test_loss:.4f}  MAE={test_mae:.4f} (days)")
    print(f"Checkpoint: {save_dir / ckpt_name}")
    return model, info


if __name__ == "__main__":
    main(epochs=100, lr=1e-4, hidden_dim=32, num_hidden=2, run_name="2layer")
