import argparse
import json
import uvicorn
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from loguru import logger
from openai import AsyncOpenAI

from openai_api_protocol import CompletionRequest, ChatCompletionRequest

client = AsyncOpenAI(
    base_url="http://49.235.138.227:8080/v1",
    api_key="sk-xxx",
)

app = FastAPI()

@app.get("/ping")
async def root():
    return "pong"

async def stream_to_generator(stream):
    async for chunk in stream:
        chunk_dict = chunk.dict(exclude_unset=True)
        json_chunk = json.dumps(chunk_dict, ensure_ascii=False)
        yield f"data: {json_chunk}\n\n"
    yield "data: [DONE]\n\n"

@app.post("/v1/completions")
async def completions(request: CompletionRequest):
    logger.info("receive completion request, stream: {}, prompt: {}".format(request.stream, request.prompt))
    stream = request.stream
    if stream is None or not stream:
        return client.completions.create(model=request.model, prompt=request.prompt,
                                         max_tokens=request.max_tokens, temperature=request.temperature,
                                         stop=request.stop, stream=False)
    else:
        stream = await client.completions.create(model=request.model, prompt=request.prompt,
                                         max_tokens=request.max_tokens, temperature=request.temperature,
                                         stop=request.stop, stream=True)
        generator = stream_to_generator(stream)
        return StreamingResponse(generator, media_type="text/event-stream")


@app.post("/v1/chat/completions")
async def chat(request: ChatCompletionRequest):
    logger.info("receive chat request, stream: {}, messages: {}".format(request.stream, request.messages))
    stream = request.stream
    if stream is None or not stream:
        return client.chat.completions.create(model=request.model, messages=request.messages,
                                              max_tokens=request.max_tokens, temperature=request.temperature,
                                              stop=request.stop, stream=False)
    else:
        stream = await client.chat.completions.create(model=request.model, messages=request.messages,
                                                max_tokens=request.max_tokens, temperature=request.temperature,
                                                stop=request.stop, stream=True)
        generator = stream_to_generator(stream)
        return StreamingResponse(generator, media_type="text/event-stream")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    uvicorn.run(app, host=args.host, port=args.port, log_level="debug")
