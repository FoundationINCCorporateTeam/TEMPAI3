from huggingface_hub import InferenceClient
from flask import Flask, request, jsonify

client = InferenceClient("HuggingFaceH4/zephyr-7b-beta")

app = Flask(__name__)

@app.route('/respond', methods=['POST'])
def respond():
    data = request.json
    message = data.get('message')
    history = data.get('history', [])
    system_message = data.get('system_message')
    max_tokens = data.get('max_tokens', 512)
    temperature = data.get('temperature', 0.7)
    top_p = data.get('top_p', 0.9)

    messages = [{"role": "system", "content": system_message}]
    for val in history:
        if val[0]:
            messages.append({"role": "user", "content": val[0]})
        if val[1]:
            messages.append({"role": "assistant", "content": val[1]})
    messages.append({"role": "user", "content": message})

    response = ""
    for message in client.chat_completion(
        messages,
        max_tokens=max_tokens,
        stream=True,
        temperature=temperature,
        top_p=top_p,
    ):
        token = message.choices[0].delta.content
        response += token

    return jsonify({"response": response})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7860)
