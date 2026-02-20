"""
Analysis mapping Olist data to the operations statement:
  "Margins squeezed, sellers complaining, bad reviews, don't know where to focus.
   What would actually make a difference?"
Run: source ~/base/bin/activate && python analysis_ops_statement.py
"""
import os
BASE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("MPLCONFIGDIR", os.path.join(BASE, ".mplcache"))
os.environ.setdefault("MPLBACKEND", "Agg")
import pandas as pd
import numpy as np
from pathlib import Path

DATA = Path(__file__).parent / "data"
OUT = Path(__file__).parent / "experiments" / "graphs"
OUT.mkdir(parents=True, exist_ok=True)

def load_and_join():
    orders = pd.read_csv(DATA / "olist_orders_dataset.csv", parse_dates=[
        "order_purchase_timestamp", "order_approved_at", "order_delivered_carrier_date",
        "order_delivered_customer_date", "order_estimated_delivery_date"
    ])
    items = pd.read_csv(DATA / "olist_order_items_dataset.csv")
    items["shipping_limit_date"] = pd.to_datetime(items["shipping_limit_date"])
    reviews = pd.read_csv(DATA / "olist_order_reviews_dataset.csv", parse_dates=["review_creation_date", "review_answer_timestamp"])
    customers = pd.read_csv(DATA / "olist_customers_dataset.csv")
    products = pd.read_csv(DATA / "olist_products_dataset.csv")
    payments = pd.read_csv(DATA / "olist_order_payments_dataset.csv")
    cat_trans = pd.read_csv(DATA / "product_category_name_translation.csv")
    sellers = pd.read_csv(DATA / "olist_sellers_dataset.csv")

    # Delivered orders with delivery timing
    delivered = orders[orders["order_status"] == "delivered"].copy()
    delivered = delivered.dropna(subset=["order_delivered_customer_date", "order_estimated_delivery_date"])
    delivered["delivery_days"] = (delivered["order_delivered_customer_date"] - delivered["order_approved_at"]).dt.total_seconds() / (24 * 3600)
    delivered["estimated_days"] = (delivered["order_estimated_delivery_date"] - delivered["order_approved_at"]).dt.total_seconds() / (24 * 3600)
    delivered["delay_days"] = delivered["delivery_days"] - delivered["estimated_days"]
    delivered["is_late"] = delivered["delay_days"] > 0

    # Order-level revenue/freight from items
    order_agg = items.groupby("order_id").agg(
        revenue=("price", "sum"),
        freight=("freight_value", "sum"),
        n_items=("order_item_id", "count"),
    ).reset_index()
    delivered = delivered.merge(order_agg, on="order_id", how="left")
    pay_tot = payments.groupby("order_id").agg(
        payment_value=("payment_value", "sum"),
        max_installments=("payment_installments", "max"),
        n_payments=("payment_sequential", "max"),
    ).reset_index()
    delivered = delivered.merge(pay_tot, on="order_id", how="left")
    delivered = delivered.merge(customers[["customer_id", "customer_state"]], on="customer_id", how="left")

    # One review per order (take max score if multiple)
    rev_agg = reviews.groupby("order_id").agg(review_score=("review_score", "max")).reset_index()
    delivered = delivered.merge(rev_agg, on="order_id", how="left")
    delivered["bad_review"] = delivered["review_score"] <= 2

    # Products -> category English
    products_cat = products.merge(cat_trans, on="product_category_name", how="left")
    items_cat = items.merge(products_cat[["product_id", "product_category_name_english"]], on="product_id", how="left")

    # Seller-level: join items -> order -> delivered + review
    items_with_order = items.merge(delivered[["order_id", "is_late", "review_score", "revenue", "delay_days"]], on="order_id", how="inner")
    seller_metrics = items_with_order.groupby("seller_id").agg(
        n_orders=("order_id", "nunique"),
        total_revenue=("revenue", "sum"),
        late_rate=("is_late", "mean"),
        mean_review=("review_score", "mean"),
        mean_delay_days=("delay_days", "mean"),
    ).reset_index()

    # Category-level: use order-level (one row per order) but attribute category from items (take first/mode per order for simplicity, or aggregate by order+category then roll up)
    order_cat = items_cat.merge(delivered[["order_id", "is_late", "review_score", "revenue", "bad_review", "customer_state"]], on="order_id", how="inner")
    cat_metrics = order_cat.groupby("product_category_name_english").agg(
        revenue=("revenue", "sum"),
        n_orders=("order_id", "nunique"),
        late_rate=("is_late", "mean"),
        mean_review=("review_score", "mean"),
        bad_review_rate=("bad_review", "mean"),
    ).reset_index()

    # State-level
    state_metrics = delivered.groupby("customer_state").agg(
        revenue=("revenue", "sum"),
        n_orders=("order_id", "count"),
        late_rate=("is_late", "mean"),
        mean_review=("review_score", "mean"),
        bad_review_rate=("bad_review", "mean"),
    ).reset_index()

    return {
        "delivered": delivered,
        "orders": orders,
        "payments": payments,
        "reviews": reviews,
        "seller_metrics": seller_metrics,
        "cat_metrics": cat_metrics,
        "state_metrics": state_metrics,
        "items_with_order": items_with_order,
    }


def run_analysis():
    d = load_and_join()
    delivered = d["delivered"]
    orders = d["orders"]
    payments = d["payments"]
    reviews = d["reviews"]
    seller_metrics = d["seller_metrics"]
    cat_metrics = d["cat_metrics"]
    state_metrics = d["state_metrics"]

    print("\n" + "=" * 70)
    print("ANALYSIS: Mapping data to the operations statement")
    print("=" * 70)

    # --- 1. MARGINS (what we can see) ---
    print("\n--- 1. MARGINS (what the data can tell us) ---")
    print("We don't have marketplace fees or costs. We can only see revenue structure and pressure indicators.\n")
    freight_pct = (delivered["freight"].sum() / delivered["payment_value"].sum() * 100) if delivered["payment_value"].notna().any() else np.nan
    print(f"  Revenue (from items, delivered): {delivered['revenue'].sum():,.0f}")
    print(f"  Total payment value (delivered): {delivered['payment_value'].sum():,.0f}")
    print(f"  Freight as % of payment (approx): {freight_pct:.1f}%  [high freight = margin pressure]")
    print(f"  Mean order value (payment): {delivered['payment_value'].mean():.1f}")
    print(f"  Mean installments: {delivered['max_installments'].mean():.1f}  [more installments = cash flow delay]")
    inst_dist = payments.groupby("order_id")["payment_installments"].max().value_counts().sort_index()
    print("  Installment distribution (max per order):")
    for k, v in inst_dist.head(12).items():
        print(f"    {k}: {v}")

    # --- 2. SELLERS (who complains) ---
    print("\n--- 2. SELLERS (who might be complaining) ---")
    print("Sellers care about: volume, timely delivery (they get blamed), and reviews.\n")
    n_sellers = len(seller_metrics)
    tot_orders = seller_metrics["n_orders"].sum()
    top10_pct_orders = seller_metrics.nlargest(10, "n_orders")["n_orders"].sum() / tot_orders * 100
    print(f"  Sellers: {n_sellers}; total order-items linked to delivered: {tot_orders}")
    print(f"  Top 10 sellers by orders: {top10_pct_orders:.1f}% of orders  [concentration]")
    late_sellers = seller_metrics[seller_metrics["late_rate"] > 0.15]
    print(f"  Sellers with >15% late rate: {len(late_sellers)} ({100*len(late_sellers)/n_sellers:.0f}%)")
    print(f"  Sellers with mean review < 4: {len(seller_metrics[seller_metrics['mean_review'] < 4])} ({100*len(seller_metrics[seller_metrics['mean_review'] < 4])/n_sellers:.0f}%)")
    print("\n  Seller stats (order-weighted):")
    print(seller_metrics.describe()[["n_orders", "late_rate", "mean_review"]].round(3).to_string())

    # --- 3. BAD REVIEWS (buyers) ---
    print("\n--- 3. BAD REVIEWS (buyers) ---")
    print("What drives low scores? Delivery delay, category, state.\n")
    score_dist = reviews["review_score"].value_counts().sort_index()
    print("  Review score distribution:")
    for s, c in score_dist.items():
        print(f"    {s}: {c} ({100*c/reviews.shape[0]:.1f}%)")
    bad_rate = delivered["bad_review"].mean() * 100
    print(f"  Bad review rate (score<=2) among delivered: {bad_rate:.1f}%")
    # Correlation: delay vs review
    late_review_low = delivered[delivered["is_late"]]["review_score"].mean()
    ontime_review = delivered[~delivered["is_late"]]["review_score"].mean()
    print(f"  Mean review when LATE: {late_review_low:.2f}; when ON-TIME: {ontime_review:.2f}")
    # By category (top problem categories)
    cat_problem = cat_metrics[cat_metrics["n_orders"] >= 100].nlargest(10, "bad_review_rate")[
        ["product_category_name_english", "n_orders", "bad_review_rate", "mean_review", "late_rate"]
    ]
    print("\n  Top 10 categories by BAD review rate (min 100 orders):")
    print(cat_problem.to_string(index=False))

    # --- 4. WHERE TO FOCUS ---
    print("\n--- 4. WHERE TO FOCUS (high impact × high pain) ---")
    print("Segments that drive revenue/volume AND have worse reviews or more late deliveries.\n")
    cat_metrics["focus_score"] = (
        cat_metrics["revenue"] / cat_metrics["revenue"].max() * 0.4
        + (1 - cat_metrics["mean_review"] / 5) * 0.35
        + cat_metrics["late_rate"] * 0.25
    )
    cat_focus = cat_metrics[cat_metrics["n_orders"] >= 200].nlargest(15, "focus_score")[
        ["product_category_name_english", "revenue", "n_orders", "mean_review", "late_rate", "bad_review_rate"]
    ]
    print("  Categories to focus (revenue + low review + late):")
    print(cat_focus.to_string(index=False))

    print("\n  States to focus (revenue + pain):")
    state_metrics["focus_score"] = (
        state_metrics["revenue"] / state_metrics["revenue"].max() * 0.4
        + (1 - state_metrics["mean_review"] / 5) * 0.3
        + state_metrics["late_rate"] * 0.3
    )
    state_focus = state_metrics.nlargest(10, "focus_score")[
        ["customer_state", "revenue", "n_orders", "mean_review", "late_rate", "bad_review_rate"]
    ]
    print(state_focus.to_string(index=False))

    # --- 5. WHAT WOULD MAKE A DIFFERENCE (summary) ---
    print("\n" + "=" * 70)
    print("5. WHAT WOULD ACTUALLY MAKE A DIFFERENCE (data-backed)")
    print("=" * 70)
    print("""
  • Margins: Data shows freight and installments (cash flow). To improve margins you'd need
    fee/cost data; ML could optimize pricing or freight. Not directly answerable from this dataset.

  • Sellers: A meaningful share of sellers have high late rates or low average reviews.
    ML that predicts "order at risk of being late" or "order at risk of bad review" would let
    you prioritise support or logistics for those orders—reducing seller blame and refunds.

  • Bad reviews: Late delivery strongly correlates with lower scores. Fixing delivery
    estimates or fulfilment would likely reduce bad reviews. ML: predict review score or
    "bad review" so you can intervene (e.g. proactive outreach, compensation) before they leave 1–2 stars.

  • Where to focus: Categories and states above are high-revenue and high-pain. Prioritise
    delivery and review improvements there first. ML can segment "at-risk" orders by category/state/seller.

  Recommended ML problems (pick 1–2):
    1. Predict bad review (binary: score <= 2) or review score (1–5) — directly addresses
       "buyers leave bad reviews" and helps target interventions.
    2. Predict late delivery (binary: delivered after estimate) — addresses seller and buyer
       pain, and "where to focus" (which orders/sellers to support).
""")

    # --- Plots ---
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))

    # A: Review score distribution
    ax = axes[0, 0]
    reviews["review_score"].value_counts().sort_index().plot(kind="bar", ax=ax, color="steelblue")
    ax.set_title("Review score distribution (all reviews)")
    ax.set_xlabel("Score")

    # B: Late vs on-time → mean review
    ax = axes[0, 1]
    comp = delivered.groupby("is_late")["review_score"].mean()
    comp.index = ["On-time", "Late"]
    comp.plot(kind="bar", ax=ax, color=["#2ecc71", "#e74c3c"])
    ax.set_title("Mean review: on-time vs late delivery")
    ax.set_ylabel("Mean review score")
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=0)

    # C: Seller concentration (orders per seller)
    ax = axes[1, 0]
    seller_metrics["n_orders"].hist(bins=50, ax=ax, color="coral", edgecolor="white")
    ax.set_title("Orders per seller (delivered orders)")
    ax.set_xlabel("Orders per seller")

    # D: Category focus (top 8 by focus_score)
    ax = axes[1, 1]
    top_cat = cat_focus.head(8)
    x = range(len(top_cat))
    ax.barh(x, top_cat["revenue"] / 1e6, alpha=0.7, label="Revenue (M)")
    ax2 = ax.twiny()
    ax2.barh(x, top_cat["bad_review_rate"] * 100, alpha=0.4, color="red", label="Bad review %")
    ax.set_yticks(x)
    ax.set_yticklabels(top_cat["product_category_name_english"], fontsize=8)
    ax.set_xlabel("Revenue (M)")
    ax2.set_xlabel("Bad review %")
    ax.set_title("Focus categories: revenue vs bad review rate")
    plt.tight_layout()
    plt.savefig(OUT / "ops_statement_analysis.png", dpi=120)
    plt.close()

    print(f"\nPlot saved: {OUT / 'ops_statement_analysis.png'}")


if __name__ == "__main__":
    run_analysis()
