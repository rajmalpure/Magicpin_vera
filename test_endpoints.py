import requests
import json

base_url = "http://localhost:8080"

print("--- Testing /v1/healthz ---")
resp = requests.get(f"{base_url}/v1/healthz")
print(resp.status_code, resp.text)

print("\n--- Testing /v1/metadata ---")
resp = requests.get(f"{base_url}/v1/metadata")
print(resp.status_code, resp.json())

print("\n--- Testing /v1/context (Category) ---")
cat_payload = {
    "type": "category",
    "id": "test_category",
    "data": {"name": "Test Category", "voice": "Friendly"}
}
resp = requests.post(f"{base_url}/v1/context", json=cat_payload)
print(resp.status_code, resp.json())

print("\n--- Testing /v1/context (Merchant) ---")
merch_payload = {
    "type": "merchant",
    "id": "test_merchant",
    "data": {"name": "Test Merchant", "category_id": "test_category"}
}
resp = requests.post(f"{base_url}/v1/context", json=merch_payload)
print(resp.status_code, resp.json())

print("\n--- Testing /v1/tick ---")
tick_payload = {
    "trigger_id": "test_trigger",
    "merchant_id": "test_merchant",
    "type": "outreach",
    "context": {"goal": "Say hello to customer"}
}
resp = requests.post(f"{base_url}/v1/tick", json=tick_payload)
print(resp.status_code, json.dumps(resp.json(), indent=2))
