import sys
import os
import time
import json

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.dashboard.api_client import api_request

def run_test():
    print("=== STARTING AUDIT FLOW TEST ===")
    
    # 1. Fetch Models
    print("\n1. Fetching Models...")
    models = api_request("/models")
    if "error" in models:
        print(f"FAILED to fetch models: {models['error']}")
        return False
    
    gpt4 = next((m for m in models.get('models', []) if m['id'] == 'openai-gpt4'), None)
    if not gpt4:
        print("WARNING: GPT-4 model not found, using first available")
        gpt4 = models['models'][0]
    
    print(f"   Selected Model: {gpt4['name']} ({gpt4['id']})")
    
    # 2. Fetch Policies
    print("\n2. Fetching Policies...")
    policies = api_request("/policies")
    if "error" in policies:
        print(f"FAILED to fetch policies: {policies['error']}")
        return False
    
    policy_ids = [p['id'] for p in policies.get('policies', [])[:2]]
    print(f"   Selected Policies: {policy_ids}")

    # 3. Launch Audit
    print("\n3. Launching Audit...")
    audit_data = {
        "model_id": gpt4['id'],
        "policy_ids": policy_ids,
        "test_count": 5,
        "frameworks": ["aura-native"]
    }
    
    response = api_request("/audit", method="POST", data=audit_data)
    if "error" in response:
        print(f"FAILED to launch audit: {response['error']}")
        return False
        
    audit_id = response.get('audit_id')
    print(f"   Audit Launched! ID: {audit_id}")
    print(f"   Initial Status: {response.get('status')}")
    
    # 4. Monitor Progress
    print("\n4. Monitoring Progress...")
    max_retries = 30
    for i in range(max_retries):
        time.sleep(2)
        # We don't have a direct status endpoint in api_client exposed easily without full url construction
        # But dashboard uses /audits or result returned.
        # Actually api_client handles base url.
        # Let's check audit listing to find our audit
        
        # NOTE: The dashboard flow waits for the POST request to return the FINAL result because the API might be synchronous 
        # or the client waits? 
        # Looking at audit_wizard.py: response = api_request("/audit", ...)
        # It seems it waits for the result?
        # Let's check if the response ALREADY contains results.
        
        if response.get('status') == 'completed':
            print("   Audit completed immediately (synchronous?)")
            break
            
        # If it's async, we'd poll. But typically this simple demo might be sync.
        # Let's inspect the response keys.
        if 'compliance_score' in response:
            print("   Results present in initial response.")
            break
            
    print("\n5. Verifying Results...")
    print(f"   Compliance Score: {response.get('compliance_score')}")
    print(f"   Total Tests: {response.get('results', {}).get('total_tests')}")
    
    if response.get('compliance_score') is not None:
         print("\n=== TEST PASSED: Full flow completed successfully ===")
         return True
    else:
         print("\n=== TEST FAILED: No results returned ===")
         return False

if __name__ == "__main__":
    success = run_test()
    sys.exit(0 if success else 1)
