from flask import Flask, render_template, request, jsonify
from huggingface_hub import InferenceClient

app = Flask(__name__)
client = InferenceClient("HuggingFaceH4/zephyr-7b-beta")

def get_response(message, history, system_message, max_tokens, temperature, top_p):
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
        if message.choices and len(message.choices) > 0:
            token = message.choices[0].delta.content
            response += token

    return response


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    message = data['message']
    history = data.get('history', [])
    system_message = data.get('system_message', "You are a friendly Chatbot.")
    max_tokens = data.get('max_tokens', 512)
    temperature = data.get('temperature', 1.0)
    top_p = data.get('top_p', 1.0)

    response = get_response(message, history, system_message, max_tokens, temperature, top_p)
    return jsonify({'response': response})

if __name__ == "__main__":
    app.run(debug=True)
