from flask import Flask, render_template, request, jsonify
import requests
import json

app = Flask(__name__)

API_URL = "https://api-inference.huggingface.co/models/HuggingFaceH4/zephyr-7b-beta"
HEADERS = {"Authorization": "Bearer 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"}

def query(payload):
    response = requests.post(API_URL, headers=HEADERS, json=payload)
    return response.json()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    message = data['message']

    payload = {"inputs": message}
    response = query(payload)
    
    # Convert the JSON response to a string before returning
    response_str = json.dumps(response)
    
    return jsonify({'response': response_str})

if __name__ == "__main__":
    app.run(debug=True)
