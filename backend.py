from fastapi import FastAPI, status, HTTPException
from pydantic import BaseModel
from huggingface_hub import InferenceClient
import os
from pathlib import Path
from dotenv import load_dotenv
import logging

# load variables from .env if present (makes it easier in dev)
env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    load_dotenv(env_path)
song_app = FastAPI()

# get hf token from env
HF_TOKEN = os.getenv("HF_TOKEN")

# warn on startup if the HF_TOKEN env variable is absent
if HF_TOKEN is None:
    logging.warning("HF_TOKEN is not set; remote inference requests will return an error.")
else:
    logging.info("HF_TOKEN loaded from environment")

# cache local pipeline
pipe = None

class UserQuery(BaseModel):
    prompt: str
    system_message: str
    max_tokens: int
    temp: float
    top_p: float
    use_local_model: bool
    hf_token: str | None = None

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

@song_app.get("/")
def read_root():
    return {"message": "Song backend running"}

@song_app.post("/generate", status_code=status.HTTP_200_OK)
def generate_song(userQuery: UserQuery):
    global pipe

    try:
        messages = [
            {"role": "system", "content": userQuery.system_message},
            {"role": "user", "content": EXAMPLE_INPUT_1},
            {"role": "assistant", "content": EXAMPLE_OUTPUT_1},
            {"role": "user", "content": EXAMPLE_INPUT_2},
            {"role": "assistant", "content": EXAMPLE_OUTPUT_2},
            {"role": "user", "content": EXAMPLE_INPUT_3},
            {"role": "assistant", "content": EXAMPLE_OUTPUT_3},
            {"role": "user", "content": userQuery.prompt}
        ]

        # local model
        if userQuery.use_local_model:
            try:
                from transformers import pipeline
                import torch
            except Exception as e:
                logging.error(f"Failed to import local model dependencies: {e}")
                raise HTTPException(
                    status_code=500,
                    detail=(
                        f"Local model unavailable: failed to import dependencies. "
                        f"Error: {str(e)[:200]}"
                    ),
                )

            try:
                if pipe is None:
                    pipe = pipeline(
                        "text-generation",
                        model="LiquidAI/LFM2-350M", 
                    )
            except Exception as e:
                logging.error(f"Failed to create local pipeline: {e}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Could not initialize local pipeline: {str(e)[:200]}",
                )

            try:
                prompt = messages
                prompt= ''.join([f"{m['role']}: {m['content']}" for m in prompt])
                outputs = pipe(
                    prompt,
                    max_new_tokens=userQuery.max_tokens,
                    do_sample=True,
                    temperature=userQuery.temp,
                    top_p=userQuery.top_p,
                )

                # Just output the answer, remove the prompt
                response = outputs[0]['generated_text'][len(prompt):].strip()
                return {"response": response}
            except Exception as e:
                logging.error(f"Local model generation error: {e}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Generation failed: {str(e)[:200]}",
                )

        else:
            # choose which token to use: explicit hf_token from request takes priority
            token_to_use = HF_TOKEN
            logging.debug(f"remote branch invoked; token present={token_to_use is not None}")
            if token_to_use is None:
                # no token available, return message without aborting connection
                return {"response": "Huggingface token required for remote model (set HF_TOKEN env var)"}

            # use huggingface client and define model
            client = InferenceClient(token=token_to_use, model="openai/gpt-oss-20b")
            try:
                completion = client.chat_completion(
                    messages,
                    max_tokens=userQuery.max_tokens,
                    temperature=userQuery.temp,
                    top_p=userQuery.top_p,
                )
            except Exception as e:
                # log exception and return HTTP 502
                logging.error(f"Inference client error: {e}")
                raise HTTPException(status_code=502, detail=f"Inference API error: {exc}")

            return {"response": completion.choices[0].message.content}

    except HTTPException:
        # re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        # catch any unexpected exceptions and return a proper error response
        logging.error(f"Unexpected error in generate_song: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error: {str(e)[:200]}",
        )
