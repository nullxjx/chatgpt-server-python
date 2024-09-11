import argparse
import uvicorn
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from loguru import logger
from openai import AsyncOpenAI

from openai_api_protocol import CompletionRequest, ChatCompletionRequest
from postprocess import stream_to_generator

client = AsyncOpenAI(
    base_url="http://49.235.138.227:8080/v1",
    api_key="sk-xxx",
)

app = FastAPI()

@app.get("/ping")
async def root():
    return "pong"


@app.post("/v1/completions")
async def completions(request: CompletionRequest):
    logger.info(f"receive completion request, stream: {request.stream}, prompt: {request.prompt}, stop: {request.stop}")
    stream = request.stream
    if stream is None or not stream:
        return client.completions.create(model=request.model, prompt=request.prompt, n=request.n,
                                         max_tokens=request.max_tokens, temperature=request.temperature,
                                         stream=False, stop=request.stop)
    else:
        stream = await client.completions.create(model=request.model, prompt=request.prompt, n=request.n,
                                                 max_tokens=request.max_tokens, temperature=request.temperature,
                                                 stream=True)
        generator = stream_to_generator(stream, request.stop)
        return StreamingResponse(generator, media_type="text/event-stream")


@app.post("/v1/chat/completions")
async def chat(request: ChatCompletionRequest):
    logger.info(f"receive completion request, stream: {request.stream}, messages: {request.messages}, stop: {request.stop}")
    stream = request.stream
    if stream is None or not stream:
        return client.chat.completions.create(model=request.model, messages=request.messages, n=request.n,
                                              max_tokens=request.max_tokens, temperature=request.temperature,
                                              stream=False, stop=request.stop)
    else:
        stream = await client.chat.completions.create(model=request.model, messages=request.messages, n=request.n,
                                                      max_tokens=request.max_tokens, temperature=request.temperature,
                                                      stream=True)
        generator = stream_to_generator(stream, request.stop)
        return StreamingResponse(generator, media_type="text/event-stream")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    uvicorn.run(app, host=args.host, port=args.port, log_level="debug")
