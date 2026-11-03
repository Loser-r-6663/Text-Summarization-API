import requests

BASE_URL = "http://127.0.0.1:8000"

def test_endpoints():
    print("=== 1. Test Endpoint GET / ===")
    res = requests.get(f"{BASE_URL}/")
    print(f"Status Code: {res.status_code}")
    print(f"Response: {res.json()}\n")

    print("=== 2. Test Endpoint GET /health ===")
    res = requests.get(f"{BASE_URL}/health")
    print(f"Status Code: {res.status_code}")
    print(f"Response: {res.json()}\n")

    print("=== 3. Test Endpoint POST /predict (Hợp lệ) ===")
    sample_text = (
        "Artificial intelligence is transforming industries across the global economy. "
        "From healthcare to finance, machine learning algorithms are automating processes, "
        "analyzing vast amounts of data, and helping professionals make better decisions. "
        "As models become more sophisticated, the focus is shifting toward safe and ethical deployment."
    )
    payload = {
        "text": sample_text,
        "max_length": 60,
        "min_length": 15
    }
    res = requests.post(f"{BASE_URL}/predict", json=payload)
    print(f"Status Code: {res.status_code}")
    print(f"Response: {res.json()}\n")

    print("=== 4. Test Endpoint POST /predict (Lỗi Input rỗng / ngắn) ===")
    invalid_payload = {"text": "Short text"}
    res = requests.post(f"{BASE_URL}/predict", json=invalid_payload)
    print(f"Status Code: {res.status_code}")
    print(f"Response: {res.json()}\n")

if __name__ == "__main__":
    test_endpoints()