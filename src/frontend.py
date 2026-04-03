import gradio as gr
import requests

LOCAL_URL = "http://localhost:9003"
BASE_URL = "http://paffenroth-23.dyn.wpi.edu:9003"

def respond(
        message,
        history,
        sys_message,
        max_tokens,
        temp,
        top_p,
        use_local_model,
        hf_token=None
):
    payload = {
        "prompt": message,
        "system_message": sys_message,
        "max_tokens": max_tokens,
        "temp": temp,
        "top_p": top_p,
        "use_local_model": use_local_model,
        "hf_token": hf_token
    }

    try:
        # post to appropriate server (local always works during development)
        #url = LOCAL_URL if use_local_model else BASE_URL
        url = BASE_URL
        response = requests.post(f"{url}/generate", json=payload, timeout=60)
        
        try:
            result = response.json()
        except ValueError:
            # backend may have crashed and returned non-JSON content
            return (
                f"Backend crashed or returned invalid response ({response.status_code}). "
                f"Check the backend logs. Response: {response.text[:200]}"
            )

        if response.status_code != 200:
            # attempt to retrieve error message from detail or response
            detail = result.get("detail") or result.get("error") or str(result)
            return f"Backend error ({response.status_code}): {detail}"

        # normal path
        return result.get("response", f"No 'response' field returned: {result}")
    except requests.exceptions.ConnectionError as e:
        return (
            f"Connection error: cannot reach backend at {url}. "
            f"Make sure the backend is running on port 9003. Error: {str(e)[:100]}"
        )
    except requests.exceptions.Timeout:
        return (
            "Backend request timed out after 60 seconds. "
            "The model may be slow or the backend is stuck."
        )
    except Exception as e:
        return f"Error connecting to backend: {e}"
    
chatbot = gr.ChatInterface(
    respond,
    additional_inputs=[
        gr.Textbox(value="You are a professional Songwriter and Lyricist." \
        " Your goal is to write lyrics that have a strong rhythm, clear structure, and creative rhymes.",
        label="System message"),
        gr.Slider(minimum=1, maximum=2048, value=512, step=1, label="Max new tokens"),
        gr.Slider(minimum=0.1, maximum=4.0, value=0.7, step=0.1, label="Temperature"),
        gr.Slider(minimum=0.1, maximum=1.0, value=0.95, step=0.05, label="Top-p (nucleus sampling)"),
        gr.Checkbox(label="Use Local Model", value=False),
    ]
)

with gr.Blocks() as demo:
    with gr.Row():
        gr.Markdown("<h1 style='text-align: center;'> 🎵 Song Generator Chatbot 🎵</h1>")
    chatbot.render()

if __name__ == "__main__":
    demo.launch(css='styles.css')
