import requests
import time
import json
import sys

BASE_URL = "http://127.0.0.1:8080"

def run_verification():
    print("🚀 Starting End-to-End Verification via API...")

    # 1. Verify Model
    print("\n1. Verifying Model...")
    models = requests.get(f"{BASE_URL}/models").json()
    model = next((m for m in models["models"] if m["name"] == "Walkthrough Check"), None)
    if not model:
        print("❌ Model 'Walkthrough Check' not found!")
        # Create it if missing
        print("   Creating model...")
        requests.post(f"{BASE_URL}/models", json={
            "name": "Walkthrough Check",
            "model_type": "api",
            "provider": "custom", 
            "model_name": "chk-001",
            "endpoint_url": "http://127.0.0.1:8080" # Self-test loopback
        })
        model = requests.get(f"{BASE_URL}/models").json()["models"][-1]
    
    model_id = model["id"]
    print(f"✅ Model found: {model['name']} ({model_id})")

    # 2. Verify Policy
    print("\n2. Verifying Policy...")
    policies = requests.get(f"{BASE_URL}/policies").json()
    policy = next((p for p in policies["policies"] if p["name"] == "NIST AI RMF"), None)
    
    if not policy:
        print("❌ Policy 'NIST AI RMF' not found! Seeding...")
        requests.post(f"{BASE_URL}/policies", json={
            "id": "compliance-standard-001",
            "name": "NIST AI RMF",
            "description": "NIST AI RMF",
            "category": "Standard",
            "rules": ["Map", "Measure", "Manage"],
            "version": "1.0.0",
            "active": True
        })
        policy = requests.get(f"{BASE_URL}/policies").json()["policies"][-1]

    policy_id = policy["id"]
    print(f"✅ Policy found: {policy['name']} ({policy_id})")

    # 3. Trigger Audit
    print("\n3. Triggering Audit...")
    payload = {
        "model_id": model_id,
        "policy_ids": [policy_id],
        "test_count": 5,
        "frameworks": ["aura-native"]
    }
    
    start_time = time.time()
    response = requests.post(f"{BASE_URL}/audit", json=payload)
    
    if response.status_code != 200:
        print(f"❌ Failed to start audit: {response.text}")
        sys.exit(1)
        
    result = response.json()
    print(f"✅ Audit started! ID: {result.get('audit_id')}")
    
    # 4. Validate Result
    print("\n4. Validating Result...")
    print(f"   Status: {result.get('status')}")
    print(f"   Compliance Score: {result.get('compliance_score')}")
    print(f"   Issues Found: {len(result.get('issues', []))}")
    
    if result.get('status') == "completed":
        print("\n✨ SUCCESS: End-to-End Audit Flow Verified!")
    else:
        print("\n⚠️ Audit did not complete successfully.")

if __name__ == "__main__":
    try:
        run_verification()
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
