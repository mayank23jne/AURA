import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.dashboard.api_client import api_request, API_URL

print(f"Checking for policies at: {API_URL}")

try:
    response = api_request("/policies")
    if "error" in response:
        print(f"FAILED to fetch policies: {response['error']}")
        sys.exit(1)
    
    policies = response.get('policies', [])
    print(f"Policies found: {len(policies)}")
    
    if len(policies) == 0:
        print("WARNING: No policies returned from backend. This explains the empty screenshot.")
        # Attempt to trigger policy initialization if possible, or just report it.
    else:
        for p in policies:
            print(f"- {p.get('name')} ({p.get('id')})")
            
except Exception as e:
    print(f"CRITICAL ERROR: {str(e)}")
    sys.exit(1)
