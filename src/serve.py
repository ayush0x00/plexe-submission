"""
FastAPI server to host the delivery-date NN. Loads checkpoint and exposes /predict.
Run: source ~/base/bin/activate && uvicorn src.serve:app --reload --host 0.0.0.0 --port 8000
"""
from pathlib import Path
import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.delivery_model import build_model
from src.delivery_dataset import FEATURE_COLS

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_DIR = PROJECT_ROOT / "experiments"
DEFAULT_CKPT = "delivery_date_nn_4layer.pt"


def _get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


DEVICE = _get_device()
app = FastAPI(title="Delivery date prediction API", version="0.1.0")

# Loaded at startup
_model = None
_scaler = None
_feature_cols = None
_delivery_days_clip = None


class PredictRequest(BaseModel):
    """One row of features in the same order as training."""

    estimated_days: float = Field(..., description="Estimated delivery days from approval")
    approval_delay_hours: float = Field(..., description="Hours from purchase to approval")
    purchase_weekday: int = Field(..., ge=0, le=6, description="0=Monday, 6=Sunday")
    purchase_hour: int = Field(..., ge=0, le=23)
    purchase_month: int = Field(..., ge=1, le=12)
    total_price: float = Field(..., ge=0)
    total_freight: float = Field(..., ge=0)
    n_items: int = Field(..., ge=1)
    n_sellers: int = Field(..., ge=1)
    shipping_slack_days: float = Field(..., description="Days from approval to shipping limit")
    total_weight_g: float = Field(..., ge=0)
    seller_mean_delivery_days: float = Field(..., ge=0)
    seller_late_rate: float = Field(..., ge=0, le=1)
    same_state: int = Field(..., ge=0, le=1)
    category_encoded: int = Field(..., ge=0)
    customer_state_code: int = Field(..., ge=0)
    seller_state_code: int = Field(..., ge=0)


class TopFeature(BaseModel):
    feature: str
    contribution: float


class PredictResponse(BaseModel):
    prediction_delivery_days: float
    top_contributing_features: list[TopFeature]


def load_model(ckpt_path: Path):
    global _model, _scaler, _feature_cols, _delivery_days_clip
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    n_features = ckpt["n_features"]
    _feature_cols = ckpt["feature_cols"]
    _scaler = ckpt["scaler"]
    _delivery_days_clip = ckpt.get("delivery_days_clip")
    _model = build_model(
        input_dim=n_features,
        hidden_dim=ckpt["hidden_dim"],
        num_hidden=ckpt["num_hidden"],
        dropout=0.0,
        device=DEVICE,
    )
    _model.load_state_dict(ckpt["model_state_dict"])
    _model.eval()
    return _model


@app.on_event("startup")
def startup():
    ckpt = CHECKPOINT_DIR / DEFAULT_CKPT
    if not ckpt.exists():
        for alt_name in ("delivery_date_nn.pt", "delivery_date_nn_2layer.pt"):
            alt = CHECKPOINT_DIR / alt_name
            if alt.exists():
                ckpt = alt
                break
    if ckpt.exists():
        load_model(ckpt)
        print(f"Loaded model from {ckpt}")
    else:
        print(f"No checkpoint at {ckpt}; /predict will return 503 until model is trained.")


@app.get("/")
def root():
    return {"service": "delivery-date-prediction", "status": "ok", "model_loaded": _model is not None}


@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": _model is not None}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if _model is None or _scaler is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Train first and restart.")
    # Build feature vector in FEATURE_COLS order
    vec = [
        req.estimated_days,
        req.approval_delay_hours,
        req.purchase_weekday,
        req.purchase_hour,
        req.purchase_month,
        req.total_price,
        req.total_freight,
        req.n_items,
        req.n_sellers,
        req.shipping_slack_days,
        req.total_weight_g,
        req.seller_mean_delivery_days,
        req.seller_late_rate,
        req.same_state,
        req.category_encoded,
        req.customer_state_code,
        req.seller_state_code,
    ]
    X = _scaler.transform([vec])
    x_t = torch.tensor(X, dtype=torch.float32, device=DEVICE)
    with torch.no_grad():
        pred = _model(x_t).item()
    if _delivery_days_clip is not None:
        low, high = _delivery_days_clip
        pred = max(low, min(high, pred))
    # Top contributing features (gradient * input)
    x_t.requires_grad_(True)
    _model.zero_grad()
    out = _model(x_t)
    out.backward()
    grad = x_t.grad.detach().squeeze().cpu().numpy()
    contrib = grad * X.squeeze()
    order = np.argsort(np.abs(contrib))[::-1]
    top = [
        TopFeature(feature=_feature_cols[i], contribution=round(float(contrib[i]), 4))
        for i in order[:5]
    ]
    return PredictResponse(prediction_delivery_days=round(pred, 4), top_contributing_features=top)


# Optional: reload checkpoint without restart
@app.post("/reload")
def reload(ckpt_name: str = DEFAULT_CKPT):
    ckpt = CHECKPOINT_DIR / ckpt_name
    if not ckpt.exists():
        raise HTTPException(status_code=404, detail=f"Checkpoint not found: {ckpt}")
    load_model(ckpt)
    return {"loaded": str(ckpt)}
