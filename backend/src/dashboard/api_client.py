import os
import httpx
import streamlit as st

# Configuration
API_URL = os.getenv("API_URL", "http://localhost:8080")

def api_request(endpoint: str, method: str = "GET", data: dict = None) -> dict:
    """Make API request to AURA backend."""
    try:
        # Longer timeout for audit endpoints (can take 2-5 minutes with PyRIT)
        timeout = 300.0 if endpoint == "/audit" else 30.0

        with httpx.Client(timeout=timeout, trust_env=False) as client:
            url = f"{API_URL}{endpoint}"
            if method == "GET":
                response = client.get(url)
            elif method == "POST":
                response = client.post(url, json=data)
            elif method == "PUT":
                response = client.put(url, json=data)
            elif method == "DELETE":
                response = client.delete(url)
            else:
                return {"error": f"Unsupported method: {method}"}

            print(f"📡 API Request: {method} {url}")
            if response.status_code == 200:
                print(f"✅ API Success: {endpoint}")
                return response.json()
            else:
                print(f"❌ API Fail: {response.status_code} - {response.text}")
                return {"error": f"API error: {response.status_code}", "detail": response.text}
    except Exception as e:
        print(f"❌ API Connection Error to {url}: {e}")
        return {"error": str(e)}
