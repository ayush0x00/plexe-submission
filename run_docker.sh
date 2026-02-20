#!/usr/bin/env bash
# Verify Docker image: build, run, hit /health and /predict, then stop.
# Usage: from repo root, ./scripts/verify_docker.sh
# Requires: Docker daemon running; experiments/ with at least one .pt (or /predict returns 503)

set -e
cd "$(dirname "$0")"
IMAGE=delivery-date-api
CONTAINER=delivery-date-api-verify

echo "Building image..."
docker build -t "$IMAGE" .

echo "Starting container..."
docker rm -f "$CONTAINER" 2>/dev/null || true
docker run -d -p 8000:8000 --name "$CONTAINER" "$IMAGE"

echo "Waiting for server..."
for i in {1..15}; do
  if curl -sf http://localhost:8000/health >/dev/null 2>&1; then break; fi
  if [[ $i -eq 15 ]]; then echo "Server did not become ready"; docker logs "$CONTAINER"; exit 1; fi
  sleep 1
done

echo "GET /health:"
curl -s http://localhost:8000/health | python3 -m json.tool

echo ""
echo "POST /predict (inference):"
PREDICT_RESP=$(curl -s -w "\n%{http_code}" -X POST "http://localhost:8000/predict" \
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
  }')
HTTP_CODE=$(echo "$PREDICT_RESP" | tail -n1)
BODY=$(echo "$PREDICT_RESP" | sed '$d')
echo "HTTP $HTTP_CODE"
echo "$BODY" | python3 -m json.tool 2>/dev/null || echo "$BODY"

docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
echo ""
if [[ "$HTTP_CODE" == "200" ]]; then
  echo "Docker verification OK (health + predict 200)."
else
  echo "Predict returned $HTTP_CODE (200 expected if experiments/ has a .pt checkpoint)."
fi
