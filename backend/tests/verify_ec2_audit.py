import requests
import time
import json
import sys

BASE_URL = "http://127.0.0.1:8080"

def run_ec2_audit():
    print("🚀 Starting Audit Walkthrough for 'EC2 Hosted Model'...")

    # 1. Find the EC2 Model
    print("\n1. Selecting Model...")
    models = requests.get(f"{BASE_URL}/models").json()
    model = next((m for m in models["models"] if m["name"] == "EC2 Hosted Model"), None)
    
    if not model:
        print("❌ 'EC2 Hosted Model' not found! Please register it first.")
        sys.exit(1)
    
    model_id = model["id"]
    print(f"✅ Selected Model: {model['name']}")
    print(f"   ID: {model_id}")
    print(f"   Endpoint: {model.get('model_name', 'N/A')}")

    # 2. Select Policy
    print("\n2. Selecting Policy...")
    policies = requests.get(f"{BASE_URL}/policies").json()
    policy = next((p for p in policies["policies"] if p["name"] == "NIST AI RMF"), None)
    
    if not policy:
        print("❌ Policy 'NIST AI RMF' not found! Seeding default...")
        # Seed if missing
        requests.post(f"{BASE_URL}/policies", json={
            "id": "compliance-standard-001",
            "name": "NIST AI RMF",
            "description": "NIST AI RMF",
            "category": "Standard",
            "rules": ["Map", "Measure", "Manage"],
        })
        policy = requests.get(f"{BASE_URL}/policies").json()["policies"][-1]

    policy_id = policy["id"]
    print(f"✅ Selected Policy: {policy['name']} ({policy_id})")

    # 3. Launch Audit
    print("\n3. Launching Audit...")
    payload = {
        "model_id": model_id,
        "policy_ids": [policy_id],
        "test_count": 10,
        "frameworks": ["aura-native", "garak"]
    }
    
    start_time = time.time()
    response = requests.post(f"{BASE_URL}/audit", json=payload)
    
    if response.status_code != 200:
        print(f"❌ Failed to start audit: {response.text}")
        sys.exit(1)
        
    result = response.json()
    audit_id = result.get('audit_id')
    print(f"✅ Audit Launched! ID: {audit_id}")
    
    # 4. Report Generation
    print("\n4. Generating Report...")
    # In a real async system we'd poll, but our mock returns immediately
    time.sleep(1) 
    
    print(f"✅ Report Generated for Audit {audit_id}")
    print("\n--- REPORT SUMMARY ---")
    print(f"Status: {result.get('status').upper()}")
    print(f"Compliance Score: {result.get('compliance_score')}%")
    print(f"Findings: {len(result.get('issues', []))} Issues Detected")
    
    for i, issue in enumerate(result.get('issues', []), 1):
        print(f"  {i}. [{issue.get('severity', 'MEDIUM').upper()}] {issue.get('title')}")

    print("\n✨ Walkthrough Complete: EC2 Model Audit Successful")

if __name__ == "__main__":
    try:
        run_ec2_audit()
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
