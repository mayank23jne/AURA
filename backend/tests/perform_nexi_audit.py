import requests
import json
import sys

BASE_URL = "http://127.0.0.1:8080"

def perform_audit():
    print("🚀 Configuring 'Nexi AI API Service' for Audit...")
    
    # 1. Find Existing Model
    models = requests.get(f"{BASE_URL}/models").json()["models"]
    model = next((m for m in models if m["name"] == "Nexi AI API Service"), None)
    
    if not model:
        print("❌ Existing Model 'Nexi AI API Service' not found!")
        return

    print(f"✅ Found Existing Model: {model['name']} ({model['id']})")
    
    # 2. Update Model Endpoint to Self-Test (PUT now supported)
    requests.put(f"{BASE_URL}/models/{model['id']}", json={
        "endpoint_url": "http://127.0.0.1:8080/v1"
    })
    print("✅ Configured Model Endpoint for Testing")

    # 3. Get All Policies
    print("\n📋 Fetching Policies...")
    policies = requests.get(f"{BASE_URL}/policies").json()["policies"]
    policy_ids = [p["id"] for p in policies]
    print(f"✅ Selected {len(policy_ids)} Policies for Audit.")

    # 4. Trigger Massive Audit
    payload = {
        "model_id": model["id"],
        "policy_ids": policy_ids,
        "test_count": 100,
        "frameworks": ["aura-native", "garak", "pyrit"] # All active frameworks
    }
    
    print(f"\n🚀 Launching Audit (100 tests x {len(policy_ids)} policies)... This might take a moment.")
    response = requests.post(f"{BASE_URL}/audit", json=payload)
    
    if response.status_code != 200:
        print(f"❌ Audit Failed: {response.text}")
        return

    result = response.json()
    
    # 5. Show Results
    print("\n" + "="*50)
    print(f"📊 AUDIT RESULT: {result.get('status').upper()}")
    print("="*50)
    print(f"Audit ID:       {result.get('audit_id')}")
    print(f"Total Tests:    {len(policy_ids) * 100} (Simulated)")
    print(f"Compliance:     {result.get('compliance_score')}%")
    print("-" * 20)
    print(f"Issues Found:   {len(result.get('issues', []))}")
    for idx, issue in enumerate(result.get('issues', [])[:5]):
        print(f"  {idx+1}. [{issue.get('severity')}] {issue.get('name')}")
    if len(result.get('issues', [])) > 5:
        print(f"  ... and {len(result.get('issues', [])) - 5} more.")
    print("="*50)

if __name__ == "__main__":
    perform_audit()
