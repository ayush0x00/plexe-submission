# Plexe FDE Take-Home: Delivery Date Prediction

ML solution for the Olist Brazilian E-commerce dataset. We predict **delivery time (days)** so the marketplace can show accurate estimates → fewer late deliveries and bad reviews.

---

## Setup

```bash
# Environment (use your Python 3.8+ env)
source ~/base/bin/activate   # or: conda activate <env>

# Dependencies
pip install pandas numpy scikit-learn torch fastapi uvicorn pydantic matplotlib

# Data
# Place Olist CSV files in data/ (see Data source below), or run download_data.py if you have Kaggle configured.
```

**Expected layout:**
```
data/
  olist_orders_dataset.csv
  olist_order_items_dataset.csv
  olist_customers_dataset.csv
  olist_products_dataset.csv
  olist_sellers_dataset.csv
  product_category_name_translation.csv
  (optional: olist_order_payments_dataset.csv, olist_order_reviews_dataset.csv, olist_geolocation_dataset.csv)
```

---

## Problem & Rationale

**Operations ask:** *"Margins squeezed, sellers complain, buyers leave bad reviews, we don't know where to focus. What would ML actually help with?"*

- **Chosen problem:** Predict **delivery date (days)** so the platform can set an accurate “expected by” date. When the estimate is right, fewer orders feel late → fewer bad reviews and less seller blame.
- **Why this (and not only “predict late” or “predict bad review”):** One model; we get a concrete number to show; fixing the estimate addresses the main driver of bad reviews (late delivery). See `observation.md` and `data_points_for_ml.md` for EDA and feature rationale.

---

## Repo structure

| Path | Purpose |
|------|--------|
| `src/delivery_dataset.py` | Load Olist data, build 17 order-level features, 60:20:20 split, optional IQR/middle-% clip on target |
| `src/delivery_model.py` | NN: input → hidden(10) × N → 1 (configurable layers) |
| `src/train.py` | Train on `delivery_days` (MSE), early stopping, save best to `experiments/` |
| `src/evaluate_test_plot.py` | Run saved model on test set; plot actual vs predicted vs estimated; MAE |
| `src/serve.py` | FastAPI app: `/predict` returns prediction + top contributing features (gradient-based) |
| `scripts/test_predict.sh` | Sample curl for `/predict` |
| `analysis_ops_statement.py` | EDA mapping data to ops statement; writes `experiments/graphs/ops_statement_analysis.png` |
| `pca_delivery.py` | PCA on features for late delivery; importance and scree; writes `experiments/graphs/pca_delivery_*.png` |
| `observation.md` | Written observations and graph index |
| `data_points_for_ml.md` | Feature list and leakage rules for delivery-date modelling |
| `PLAN.md` | Assignment checklist and step plan |
| `experiments/` | Checkpoints (e.g. `delivery_date_nn_4layer.pt`) and `experiments/graphs/` for all plots (train/val loss, test plot, PCA, EDA) |

---

## Data & features

- **Data source:** [Brazilian E-Commerce Public Dataset by Olist (Kaggle)](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce).
- **Target:** `delivery_days` = days from `order_approved_at` to `order_delivered_customer_date` (delivered orders only).
- **Features (17):** `estimated_days`, `approval_delay_hours`, `purchase_weekday`, `purchase_hour`, `purchase_month`, `total_price`, `total_freight`, `n_items`, `n_sellers`, `shipping_slack_days`, `total_weight_g`, `seller_mean_delivery_days`, `seller_late_rate`, `same_state`, `category_encoded`, `customer_state_code`, `seller_state_code`. All known at order approval (no leakage). See `data_points_for_ml.md`.
- **Target clipping:** By default we keep the middle 75% of the training target (remove 25% extremities) via percentiles; optional fixed clip or IQR in `get_dataloaders`.

---

## Train

```bash
python -m src.train
```

- Uses 60:20:20 train/val/test (seed 42), `delivery_days` with optional IQR/middle-% clip.
- Saves best model by val loss to `experiments/delivery_date_nn_<run_name>.pt` (default `run_name="4layer"`).
- Writes `experiments/graphs/train_val_loss_<run_name>.png`.
- Early stopping when val loss > train loss for 3 consecutive epochs; max 150 epochs.

---

## Evaluate (test set & plot)

```bash
python -m src.evaluate_test_plot
```

- Loads `experiments/delivery_date_nn.pt` (or set `CHECKPOINT_PATH` in the script).
- Runs on test set; plots **actual**, **predicted**, and **estimated** delivery time (one line each).
- Saves `experiments/graphs/test_actual_vs_predicted_vs_estimated.png` and prints MAE (model vs actual, estimate vs actual).

---

## API

**Endpoints:** GET `/` (service info), GET `/health` (health check), POST `/predict` (inference), POST `/reload?ckpt_name=...` (reload checkpoint).  
On startup the app loads `experiments/delivery_date_nn_4layer.pt`, then `delivery_date_nn.pt` or `delivery_date_nn_2layer.pt`. If no checkpoint exists, `/predict` returns 503.

### Start the server

```bash
source ~/base/bin/activate
cd /path/to/Plexe
uvicorn src.serve:app --host 0.0.0.0 --port 8000
```

### Curls

**Root (service info):**
```bash
curl -s http://localhost:8000/
```

**Health check:**
```bash
curl -s http://localhost:8000/health
```

**Inference (predict delivery days):**
```bash
curl -s -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "estimated_days": 25.0,
    "approval_delay_hours": 2.5,
    "purchase_weekday": 2,
    "purchase_hour": 14,
    "purchase_month": 6,
    "total_price": 150.0,
    "total_freight": 25.0,
    "n_items": 2,
    "n_sellers": 1,
    "shipping_slack_days": 5.0,
    "total_weight_g": 1200.0,
    "seller_mean_delivery_days": 18.0,
    "seller_late_rate": 0.08,
    "same_state": 1,
    "category_encoded": 3,
    "customer_state_code": 10,
    "seller_state_code": 10
  }'
```

Example response: `{"prediction_delivery_days": 7.51, "top_contributing_features": [{"feature": "seller_mean_delivery_days", "contribution": 2.65}, ...]}`

**Reload checkpoint (e.g. after training a new model):**
```bash
curl -s -X POST "http://localhost:8000/reload?ckpt_name=delivery_date_nn_2layer.pt"
```

---

## EDA & PCA (optional)

- **Ops-focused EDA:** `python analysis_ops_statement.py` → console summary + `experiments/graphs/ops_statement_analysis.png`.  
- **PCA (importance for late delivery):** `python pca_delivery.py` → console + `experiments/graphs/pca_delivery_scree_loadings.png`, `experiments/graphs/pca_delivery_importance.png`.

---

## Metrics & evaluation

- **Training:** MSE on `delivery_days` (clipped when IQR/middle-% is used).  
- **Evaluation:** MAE (days) on test set; plot of actual vs predicted vs estimated.  
- **Limitations:** Model predicts in the (clipped) range used at train time; long-tail extremes are clipped. Use predictions as “expected days” for display; combine with business rules for guarantees.

---

## Licence & data

Code in this repo is for the Plexe FDE take-home. Olist dataset is under its own licence (see Kaggle).
