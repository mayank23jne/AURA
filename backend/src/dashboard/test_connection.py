import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.dashboard.api_client import api_request, API_URL

print(f"Testing API connection to: {API_URL}")

try:
    response = api_request("/models")
    if "error" in response:
        print(f"FAILED: {response['error']}")
        if "detail" in response:
            print(f"Detail: {response['detail']}")
        sys.exit(1)
    else:
        print("SUCCESS: Connected and fetched models.")
        print(f"Models found: {len(response.get('models', []))}")
        for m in response.get('models', []):
            print(f"- {m.get('name')} ({m.get('id')})")
        sys.exit(0)
except Exception as e:
    print(f"CRITICAL ERROR: {str(e)}")
    sys.exit(1)
