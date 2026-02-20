# Observations: Mapping data to the operations statement

> **Operations statement:**  
> "We're growing fast but our margins are getting squeezed. Sellers are complaining, buyers leave bad reviews, and we don't really know where to focus. We've heard ML can help but we don't know what to build first. Can you look at our data and tell us what would actually make a difference?"

---

## 1. Margins (what the data can tell us)

We don't have marketplace fees or cost data. We can only see revenue structure and **pressure indicators**.

| Metric | Value |
|--------|--------|
| Revenue from items (delivered) | 13,220,249 |
| Total payment value (delivered) | 15,421,083 |
| **Freight as % of payment** | **~14.3%** — high freight share implies margin pressure |
| Mean order value (payment) | 159.9 |
| Mean installments per order | 2.9 — more installments → cash flow delay |

Most orders are 1–2 installments; a large share use 6–10 installments, which delays cash flow.

**Conclusion:** Margins cannot be measured from this dataset. ML could help only with additional data (fees, costs, pricing). Freight and installment structure are the only margin-related levers visible here.

**Relevant graphs:**  
- [Revenue by category](experiments/graphs/04_revenue_by_category.png) — where revenue is concentrated  
- [Orders over time](experiments/graphs/05_orders_over_time.png) — volume trend  

---

## 2. Sellers (who might be complaining)

Sellers care about volume, timely delivery (they get blamed for late orders), and reviews.

| Metric | Value |
|--------|--------|
| Sellers (with delivered orders) | 2,970 |
| Top 10 sellers share of orders | **14.0%** (moderate concentration) |
| Sellers with **>15% late rate** | **469 (16%)** |
| Sellers with **mean review < 4** | **842 (28%)** |

Seller-level stats (delivered orders): median ~7 orders/seller, but high variance (max 1,819); median late rate 0%, but 75th percentile already at 10% late.

**Conclusion:** A meaningful share of sellers face high late rates or low average reviews. ML that predicts "order at risk of late" or "at risk of bad review" would let the platform prioritise support or logistics and reduce seller blame and refunds.

**Relevant graphs:**  
- [Ops statement analysis (4-panel)](experiments/graphs/ops_statement_analysis.png) — includes **orders per seller** distribution (bottom-left)  
- [On-time vs late (pie)](experiments/graphs/07_on_time_pie.png) — overall delivery performance sellers are judged by  

---

## 3. Bad reviews (buyers)

What drives low scores? Delivery delay, category, and geography.

| Metric | Value |
|--------|--------|
| **Mean review when LATE** | **2.57** |
| **Mean review when ON-TIME** | **4.30** |
| Bad review rate (score ≤ 2) among delivered | **12.7%** |

Review score distribution (all reviews): 1 (11.5%), 2 (3.2%), 3 (8.2%), 4 (19.3%), 5 (57.8%).

**Top categories by bad review rate** (min 100 orders): office_furniture (25.2%), fashion_male_clothing (24.8%), fixed_telephony (22.7%), audio (21.3%), home_confort (19.3%), bed_bath_table (17.8%), furniture_decor (17.7%), and others.

**Conclusion:** Late delivery strongly correlates with lower scores. Improving delivery and estimates should reduce bad reviews. ML: predict **bad review** (or score) so the platform can intervene (e.g. proactive outreach, compensation) before customers leave 1–2 stars.

**Relevant graphs:**  
- [Review score distribution](experiments/graphs/03_review_scores.png)  
- [Ops statement analysis](experiments/graphs/ops_statement_analysis.png) — **mean review: on-time vs late** (top-right)  
- [Review score by delivery delay bucket](experiments/graphs/08_review_vs_delay.png) — scores by early/on-time/late  

---

## 4. Where to focus (high impact × high pain)

Segments that drive revenue/volume **and** have worse reviews or more late deliveries.

### Categories to focus (revenue + low review + late)

| Category | Revenue | n_orders | mean_review | late_rate | bad_review_rate |
|----------|---------|----------|-------------|-----------|-----------------|
| bed_bath_table | 1.40M | 9,272 | 3.93 | 8.4% | 17.8% |
| health_beauty | 1.40M | 8,647 | 4.19 | 9.1% | 12.4% |
| computers_accessories | 1.33M | 6,529 | 3.99 | 7.8% | 16.9% |
| watches_gifts | 1.27M | 5,493 | 4.07 | 8.3% | 14.7% |
| furniture_decor | 1.11M | 6,307 | 3.96 | 8.4% | 17.7% |
| office_furniture | 0.50M | 1,254 | **3.52** | 8.9% | **25.2%** |

Office_furniture has the worst mean review and highest bad-review rate among sizable categories.

### States to focus (revenue + pain)

| State | Revenue | n_orders | mean_review | late_rate | bad_review_rate |
|-------|---------|----------|-------------|-----------|-----------------|
| SP | 5.07M | 40,494 | 4.25 | 5.9% | 10.6% |
| RJ | 1.76M | 12,350 | 3.97 | **13.5%** | **18.1%** |
| MG | 1.55M | 11,354 | 4.19 | 5.6% | 11.6% |
| BA | 0.49M | 3,256 | 3.93 | 14.0% | 16.8% |
| AL | 0.08M | 397 | 3.86 | **23.9%** | **20.9%** |

RJ and BA combine high revenue with high late and bad-review rates; AL has the worst late and bad-review rates (smaller volume).

**Conclusion:** Prioritise delivery and review improvements in the categories and states above. ML can segment "at-risk" orders by category, state, and seller.

**Relevant graphs:**  
- [Revenue by category](experiments/graphs/04_revenue_by_category.png)  
- [Revenue by state](experiments/graphs/06_revenue_by_state.png)  
- [Ops statement analysis](experiments/graphs/ops_statement_analysis.png) — **focus categories: revenue vs bad review rate** (bottom-right)  

---

## 5. What would actually make a difference (data-backed)

| Pain point | What the data shows | What ML could do |
|------------|---------------------|------------------|
| **Margins** | Only freight and installments visible; no fee/cost data | Not directly answerable; would need cost/fee data for pricing or margin optimisation |
| **Sellers** | 16% of sellers have >15% late rate; 28% have mean review &lt; 4 | Predict **order at risk of late** or **at risk of bad review** → prioritise support/logistics and reduce blame and refunds |
| **Bad reviews** | Late delivery → much lower scores (2.57 vs 4.30) | Predict **bad review** or **review score** → target at-risk orders for intervention before 1–2 stars |
| **Where to focus** | High-revenue, high-pain categories and states identified above | Segment at-risk orders by category/state/seller; prioritise operations there first |

### Recommended ML problem (one model, one lever)

**Predict delivery date / delivery delay** (regression: e.g. `delivery_days` or `delay_days`) so the platform can **show an accurate estimated delivery date**. When the estimate is right, fewer orders feel "late" to the customer → **fewer bad reviews**, less seller blame, and clearer logistics focus.

- One model instead of separate "predict late" and "predict bad review."
- Root cause: bad reviews are largely driven by late delivery; fixing the estimate fixes perception.
- Use at **order approval**: output predicted delivery date → set the shown "expected by" to this value.

*Alternative if you want a second model:* Predict **bad review** (post-delivery, before review) to target at-risk orders for intervention (e.g. proactive message). The primary lever remains better delivery-date prediction.

---

## 6. Model training and test evaluation (delivery-date NN)

**Training (2-layer NN, middle-75% target clip):**

| Item | Value |
|------|--------|
| Device | mps |
| delivery_days bounds (train, middle 75%) | (4.26, 20.71) days |
| Target after clip | min=4.26, max=20.71, mean=11.02 |
| Epoch 1 | train_loss=116.99, val_loss=55.83 |
| Early stop | Epoch 5 (val > train for 3 consecutive epochs) |
| Best checkpoint | `experiments/delivery_date_nn_2layer.pt` |
| Test (best model) | **MSE=30.70, MAE=3.67 days** |

**Test-set evaluation (same checkpoint):**

| Item | Value |
|------|--------|
| Test samples | 19,292 |
| Predictions | Clipped to (4.26, 20.71) to match training target range |
| **MAE (model prediction vs actual)** | **5.06 days** |
| **MAE (data estimate vs actual)** | **12.81 days** |

The model improves on the data’s own estimate (MAE 5.06 vs 12.81 days). Predictions are clipped to the training target range (middle 75% of train delivery_days).

**Relevant graphs:**  
- [Train vs val loss (2-layer)](experiments/graphs/train_val_loss_2layer.png)  
- [Test: actual vs predicted vs estimated](experiments/graphs/test_actual_vs_predicted_vs_estimated.png)  

---

## Graph index

| Graph | Path | Relevance |
|-------|------|-----------|
| Ops statement (4-panel) | [experiments/graphs/ops_statement_analysis.png](experiments/graphs/ops_statement_analysis.png) | Review distribution, on-time vs late review, seller concentration, focus categories |
| Order status | [experiments/graphs/01_order_status.png](experiments/graphs/01_order_status.png) | Overall order outcomes |
| Delivery delay | [experiments/graphs/02_delivery_delay.png](experiments/graphs/02_delivery_delay.png) | Distribution of delay (actual − estimated) |
| Review scores | [experiments/graphs/03_review_scores.png](experiments/graphs/03_review_scores.png) | Review score distribution |
| Revenue by category | [experiments/graphs/04_revenue_by_category.png](experiments/graphs/04_revenue_by_category.png) | Revenue concentration by category |
| Orders over time | [experiments/graphs/05_orders_over_time.png](experiments/graphs/05_orders_over_time.png) | Volume trend |
| Revenue by state | [experiments/graphs/06_revenue_by_state.png](experiments/graphs/06_revenue_by_state.png) | Geographic revenue |
| On-time vs late (pie) | [experiments/graphs/07_on_time_pie.png](experiments/graphs/07_on_time_pie.png) | Share of orders on-time vs late |
| Review vs delay | [experiments/graphs/08_review_vs_delay.png](experiments/graphs/08_review_vs_delay.png) | Mean review by delivery delay bucket |
| Train vs val loss (2-layer) | [experiments/graphs/train_val_loss_2layer.png](experiments/graphs/train_val_loss_2layer.png) | Training and validation MSE by epoch |
| Test: actual vs predicted vs estimated | [experiments/graphs/test_actual_vs_predicted_vs_estimated.png](experiments/graphs/test_actual_vs_predicted_vs_estimated.png) | Test set: actual delivery days, model prediction, data estimate |
