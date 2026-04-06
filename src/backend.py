from fastapi import FastAPI, status, HTTPException, Response
from typing import List, Optional
from pydantic import BaseModel
from huggingface_hub import InferenceClient
import prometheus_client
import os
from pathlib import Path
from dotenv import load_dotenv
import logging
from time import perf_counter

# prometheus metrics
HEALTH_REQUESTS_TOTAL = prometheus_client.Counter(
    'backend_health_requests_total',
    'Total num of /health requests received by the backend'
)
SONG_REQUESTS_TOTAL = prometheus_client.Counter(
    'backend_song_requests_total',
    'Total number of /generate requests'
)
SONG_REQUEST_ERRORS_TOTAL = prometheus_client.Counter(
    'backend_song_request_errors_total',
    'Total number of failed /song requests'
)
SONG_REQUEST_DURATION_SECONDS = prometheus_client.Histogram(
    'backend_song_request_duration_seconds',
    'Time spent generating song responses'
)

# song app
song_app = FastAPI(title="Songbird API")

# environment
HF_TOKEN = os.getenv("HF_TOKEN")
if HF_TOKEN is None:
    logging.warning("HF_TOKEN is not set.")
else:
    logging.info("HF_TOKEN loaded.")

@song_app.get("/health")
def health() -> dict[str, str]:
    HEALTH_REQUESTS_TOTAL.inc()
    return {'status': 'ok'}

# request/response models
pipe = None

class GenerateRequest(BaseModel):
    prompt: str
    system_message: str
    max_tokens: int = 200
    temp: float = 0.7
    top_p: float = 0.9
    use_local_model: bool = False
    hf_token: Optional[str] = None

# Creating example inputs and outputs for few shot learning
EXAMPLE_INPUT_1 = 'Make me lyrics and chords for a song in the style of Simon and Garfunkel about sitting through a computer science lecture'
EXAMPLE_OUTPUT_1 = """
'Fluorescent Afternoon'
Verse:
G - Em - C - G - G - Em - C - D
Morning light through dusty panes  
Coffee cooling in my hand 
The screen glows blue with ancient code 
Only half of it is planned 
The professor clears his throat 
Like a ritual we all know 
I underline a word called theory 
But I don't know where it goes 
Chorus:
C - G - Em - C - C - G - D
Hello, pointers, my old friends 
I've come to misread you again 
Because a slide was softly creeping 
Left its syntax while I was sleeping 
And the thought that formed inside my brain 
Was interrupted once again 
By the hum of fluorescent afternoon 
"""
EXAMPLE_INPUT_2 = 'Make me lyrics and chords for a song in the style of Travis Scott about someone driving to school'
EXAMPLE_OUTPUT_2 = """
'Late Bell (AM Drive)'
Hook:
Fm - Db - Ab - Eb - Fm - Db - Ab - Eb
I'm riding to school with the sun in my eyes 
Radio low but the bass still cries 
Running these lights, yeah I'm losing my time 
Late bell ringing but I'm still gonna slide 
Windows down, let the cold air bite 
Thoughts too loud in the early light 
I'm not awake but I'm still alive 
On the way to class, yeah I'm still gonna ride 
Verse:
Fm - Db - Ab - Eb - Fm - Db - Ab - Eb
Seat lean back, backpack on the floor 
Same street dreams that I had before 
Teachers talk but my mind elsewhere 
Trying find a future in the traffic glare 
Gas light on, but I'm pushing my luck 
Need more sleep, need way more trust 
Clock keep yelling that I'm behind 
But my soul moving faster than the hands of time 
"""
EXAMPLE_INPUT_3 = 'Make me chords and lyrics for a song in the style of Nirvana about Charlie Kirk'
EXAMPLE_OUTPUT_3 = """
'Campus Static' 
Verse:
Em - G - A - C - Em - G - A - C
T-shirt slogans, megaphone grin 
Selling answers in a paper-thin skin 
Talks real loud, says he's saving my soul 
But he's reading from a script he was sold 
Dorm room rage, hotel stage 
Same old war in a different age 
Says “think free” but it sounds rehearsed 
Like a bad idea wearing a tie and a curse 
Pre-chorus:
A - C - A - C
You say it's simple 
Like I'm dumb 
If I don't clap 
You say I've lost 
Chorus:
C - A - Em - G - C - A - Em - G
I don't need you 
Talking at me 
Like I'm broken 
Like I'm empty 
You don't scare me 
You just bore me 
Selling fear like 
It's conformity 
"""
FEW_SHOT_EXAMPLES = [
    {
        "input": EXAMPLE_INPUT_1,
        "output": EXAMPLE_OUTPUT_1,
    },
    {
        "input": EXAMPLE_INPUT_2,
        "output": EXAMPLE_OUTPUT_2,
    },
    {
        "input": EXAMPLE_INPUT_3,
        "output": EXAMPLE_OUTPUT_3,
    },
]

class GenerateResponse(BaseModel):
    response: str

# build prompts
def build_messages(req: GenerateRequest) -> list[dict]:
    messages = [
        {"role": "system", "content": req.system_message}
    ]

    # add few shot learning examples
    for example in FEW_SHOT_EXAMPLES:
        messages.append({"role": "user", "content": example["input"]})
        messages.append({"role": "assistant", "content": example["output"]})

    # add user prompt
    messages.append({"role": "user", "content": req.prompt})
    return messages

# local model
def generate_local(messages: list[dict], req: GenerateRequest) -> str:
    global pipe

    try:
        from transformers import pipeline
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Local model import failed: {str(e)[:200]}",
        )

    if pipe is None:
        pipe = pipeline(
            "text-generation",
            model="LiquidAI/LFM2-350M",
        )

    prompt = "".join([f"{m['role']}: {m['content']}\n" for m in messages])

    outputs = pipe(
        prompt,
        max_new_tokens=req.max_tokens,
        do_sample=True,
        temperature=req.temp,
        top_p=req.top_p,
    )

    return outputs[0]["generated_text"][len(prompt):].strip()

# remote model generation
def generate_remote(messages: list[dict], req: GenerateRequest) -> str:
    token = req.hf_token or HF_TOKEN

    if token is None:
        raise HTTPException(
            status_code=400,
            detail="HF token required for remote model",
        )

    client = InferenceClient(
        token=token,
        model="openai/gpt-oss-20b",
    )

    try:
        completion = client.chat_completion(
            messages,
            max_tokens=req.max_tokens,
            temperature=req.temp,
            top_p=req.top_p,
        )
        return completion.choices[0].message.content
    except Exception as e:
        raise HTTPException(
            status_code=502,
            detail=f"Inference API error: {str(e)[:200]}",
        )
    
# generation function
def generate(req: GenerateRequest) -> str:
    messages = build_messages(req)

    if req.use_local_model:
        return generate_local(messages, req)
    else:
        return generate_remote(messages, req)
    
@song_app.get('/health')
def health():
    return {"status": "ok"}

@song_app.post('/generate', response_model=GenerateResponse)
def generate_endpoint(body: GenerateRequest):
    SONG_REQUESTS_TOTAL.inc()
    start = perf_counter()

    try:
        song = generate(body.prompt)
        return GenerateResponse(
            response=song
        )
    except Exception as e:
        SONG_REQUEST_ERRORS_TOTAL.inc()
        logging.error(f"Generation error: {e}", exc_info=True)
        raise
    finally:
        SONG_REQUEST_DURATION_SECONDS.observe(perf_counter() - start)

@song_app.get("/metrics")
def metrics():
    return Response(
        prometheus_client.generate_latest(),
        media_type=prometheus_client.CONTENT_TYPE_LATEST,
    )
