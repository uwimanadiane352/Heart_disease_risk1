import requests
import json


API_URL = "http://127.0.0.1:5000/api/predict"

test_data = {
    "age": 55,
    "sex": "Male",
    "cp": "Typical Angina",
    "trestbps": 130,
    "chol": 250,
    "fbs": "False",
    "restecg": "Normal",
    "thalach": 150,
    "exang": "No",
    "oldpeak": 1.5,
    "slope": "Flat",
    "ca": 0.0,
    "thal": "Normal"
}

try:
    print("Sending test request to API...\n")
    
    response = requests.post(
        API_URL,
        headers={"Content-Type": "application/json"},
        data=json.dumps(test_data)
    )

    print("Status Code:", response.status_code)
    print("\nResponse JSON:")
    print(json.dumps(response.json(), indent=4))

except requests.exceptions.ConnectionError:
    print(" ERROR: Could not connect to the API.")
    print("Make sure Flask is running on http://127.0.0.1:5000")
