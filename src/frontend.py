import gradio as gr
import requests
import prometheus_client
from time import perf_counter

BACKEND_URL = 'http://backend:8000'


FRONTEND_CHAT_REQUESTS_TOTAL = prometheus_client.Counter(
    'frontend_requests_total',
    'Total number of requests',
)
FRONTEND_CHAT_REQUESTS_ERRORS_TOTAL = prometheus_client.Counter(
    'frontend_requests_errors_total',
    'Total number of errors',
)
FRONTEND_CHAT_REQUESTS_DURATION_SECONDS = prometheus_client.Histogram(
    'frontend_requests_duration_seconds',
    'Duration of frontend chat requests in seconds'
)

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
    FRONTEND_CHAT_REQUESTS_TOTAL.inc()
    started = perf_counter()
    message = message.strip()

    if not message:
        FRONTEND_CHAT_REQUESTS_ERRORS_TOTAL.inc()
        FRONTEND_CHAT_REQUESTS_DURATION_SECONDS.observe(perf_counter() - started)
        return '<p>Please enter a message.</p>'

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

        response = requests.post(f"{BACKEND_URL}/generate", json=payload, timeout=60)
        response.raise_for_status()
        result = response.json()
        return result.get("response", f"No 'response' field returned: {result}")
        
    except requests.RequestException as exc:
        FRONTEND_CHAT_REQUESTS_ERRORS_TOTAL.inc()
        return f'<p>Backend request failed: {exc}</p>'

    except ValueError:
        FRONTEND_CHAT_REQUESTS_ERRORS_TOTAL.inc()
        return "Backend returned invalid JSON."
    finally:
        FRONTEND_CHAT_REQUESTS_DURATION_SECONDS.observe(perf_counter() - started)

prometheus_client.start_http_server(8080)
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

demo.launch(server_name="0.0.0.0")
