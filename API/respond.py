from flask import Flask, request, jsonify
import requests

app = Flask(__name__)

# Define your Hugging Face API token
HF_API_TOKEN = "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"

@app.route('/respond', methods=['POST'])
def respond():
    data = request.json
    message = data.get('message')
    history = data.get('history', [])
    system_message = data.get('system_message')
    max_tokens = data.get('max_tokens', 512)
    temperature = data.get('temperature', 0.7)
    top_p = data.get('top_p', 0.9)

    # Construct the messages
    messages = [{"role": "system", "content": system_message}]
    for val in history:
        if val[0]:
            messages.append({"role": "user", "content": val[0]})
        if val[1]:
            messages.append({"role": "assistant", "content": val[1]})
    messages.append({"role": "user", "content": message})

    # Send the request to Hugging Face API
    response = requests.post(
        "https://api-inference.huggingface.co/models/HuggingFaceH4/zephyr-7b-beta",
        headers={"Authorization": f"Bearer {HF_API_TOKEN}"},
        json={"inputs": messages, "parameters": {"max_tokens": max_tokens, "temperature": temperature, "top_p": top_p}}
    )

    # Return the response
    if response.status_code == 200:
        result = response.json()
        return jsonify({"response": result['choices'][0]['text']})
    else:
        return jsonify({"error": "Failed to get response from model"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7860)
