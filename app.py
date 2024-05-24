import gradio as gr
from huggingface_hub import InferenceClient

"""
For more information on `huggingface_hub` Inference API support, please check the docs: https://huggingface.co/docs/huggingface_hub/v0.22.2/en/guides/inference
"""
client = InferenceClient("HuggingFaceH4/zephyr-7b-beta")


def respond(
    message,
    history: list[tuple[str, str]],
    system_message,
    max_tokens,
    temperature,
    top_p,
):
    messages = [{"role": "system", "content": system_message}]

    for val in history:
        if val[0]:
            messages.append({"role": "user", "content": val[0]})
        if val[1]:
            messages.append({"role": "assistant", "content": val[1]})

    messages.append({"role": "user", "content": message})

    response = ""

    try:
        for message in client.chat_completion(
            messages,
            max_tokens=max_tokens,
            stream=True,
            temperature=temperature,
            top_p=top_p,
        ):
            token = message.choices[0].delta.get("content", "")
            response += token
            yield response
    except Exception as e:
        yield f"Error: {str(e)}"

"""
For information on how to customize the ChatInterface, peruse the gradio docs: https://www.gradio.app/docs/chatinterface
"""
demo = gr.ChatInterface(
    respond,
    additional_inputs=[
        gr.Textbox(value="You are a friendly Chatbot.", label="System message", interactive=True),
        gr.Slider(minimum=1, maximum=2048, value=512, step=1, label="Max new tokens", interactive=True),
        gr.Slider(minimum=0.0, maximum=1.0, value=1.0, step=0.01, label="Temperature", interactive=True),
        gr.Slider(minimum=0.0, maximum=1.0, value=0.9, step=0.01, label="Top P", interactive=True)
    ],
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
