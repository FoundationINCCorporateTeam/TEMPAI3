import gradio as gr
from huggingface_hub import InferenceClient
from supabase import create_client, Client

# Supabase credentials
supabase_url = "https://kjrcvkmzztpjpbyfoayr.supabase.co"
supabase_key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImtqcmN2a216enRwanBieWZvYXlyIiwicm9sZSI6ImFub24iLCJpYXQiOjE3MTY3NDQwNzIsImV4cCI6MjAzMjMyMDA3Mn0.epcMnUqZONlMeCd_eUgh9opgSlZbNeVKnzARF-pHYg0"
supabase: Client = create_client(supabase_url, supabase_key)

# Hugging Face credentials
access_token = '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'
client = InferenceClient("HuggingFaceH4/zephyr-7b-beta", token=access_token)

def log_to_supabase(input_text, output_text):
    try:
        response = supabase.table('mnairecords').insert({'input': input_text, 'output': output_text}).execute()
        if response.status_code == 201:
            print("Record inserted successfully.")
        else:
            print("Failed to insert record:", response.data)
    except Exception as e:
        print("Error logging to Supabase:", e)

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
    finally:
        log_to_supabase(message, response)

"""
For information on how to customize the ChatInterface, peruse the gradio docs: https://www.gradio.app/docs/chatinterface
"""
demo = gr.ChatInterface(
    respond,
    additional_inputs=[
        gr.Textbox(value="You are a friendly Chatbot.", label="System message", interactive=True),
        gr.Slider(minimum=1, maximum=2048, value=512, step=1, label="Max new tokens", interactive=True)
    ],
    description="By using this AI, you agree to the <a href='https://school.picinel.ro/mnairedirect' target='_blank'>Terms of Service and Acceptable Use Policy</a>."
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
