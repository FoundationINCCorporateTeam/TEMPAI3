import gradio as gr
import requests

API_URL = "http://localhost:7860/api/respond"

def call_respond(message, history, system_message, max_tokens, temperature, top_p):
    response = requests.post(
        API_URL,
        json={
            "message": message,
            "history": history,
            "system_message": system_message,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
        },
    )
    return response.json()["response"]

def respond(message, history, system_message, max_tokens, temperature, top_p):
    return call_respond(message, history, system_message, max_tokens, temperature, top_p)

demo = gr.ChatInterface(
    respond,
    additional_inputs=[
        gr.Textbox(value="You are a friendly Chatbot.", label="System message"),
        gr.Slider(minimum=1, maximum=2048, value=512, step=1, label="Max new tokens"),
    ],
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
