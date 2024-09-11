import json
from stop import remove_suffix
from openai.types.completion import Completion
from openai.types.chat import ChatCompletionChunk
from openai._streaming import AsyncStream
from collections import defaultdict
from loguru import logger

async def stream_to_generator(stream: AsyncStream, stop_words=None, n=1):
    if stream._cast_to == ChatCompletionChunk:
        async for chunk in postprocess_chat(stream, stop_words, n):
            chunk_dict = chunk.dict(exclude_unset=True)
            json_chunk = json.dumps(chunk_dict, ensure_ascii=False)
            yield f"data: {json_chunk}\n\n"
        yield "data: [DONE]\n\n"
    elif stream._cast_to == Completion:
        async for chunk in postprocess_completion(stream, stop_words, n):
            chunk_dict = chunk.dict(exclude_unset=True)
            json_chunk = json.dumps(chunk_dict, ensure_ascii=False)
            yield f"data: {json_chunk}\n\n"
        yield "data: [DONE]\n\n"
    else:
        raise ValueError(f"Expected AsyncStream[Completion] or AsyncStream[ChatCompletionChunk], got {type(stream)}")

    # 注意，上边可能已经触发停止条件了，为了让模型停止推理，需要关闭这个流，模型推理服务会接收到 Aborted 请求，从而停止推理
    try:
        await stream.close()
    except Exception as e:
        logger.error(e)


async def postprocess_completion(stream: AsyncStream[Completion], stop_words=None, n=1):
    """
    有些推理框架存在 stop_words 识别不到的问题，这里加一层识别
    """
    string_buffer = defaultdict(lambda: "")  # key为 choices 数组元素中的index，value为字符串缓冲区
    choices_stop = defaultdict(lambda: False)
    async for data in stream:
        if all(choices_stop.get(i, False) is True for i in range(n)):
            logger.info(f"all choices stop")
            return
        choice = data.choices[0]
        index = choice.index
        chunk = choice.text
        if choices_stop[index]:
            continue
        remain, is_stop, stop_word = remove_suffix(string_buffer[index], chunk, stop_words)
        choice.text = remain
        if is_stop:
            choices_stop[index] = True
            choice.finish_reason= 'stop'
            logger.info(f"id {data.id}, index {index}, hit stop word: {stop_word.encode('unicode_escape').decode()}")

        string_buffer[index] += chunk
        yield data


async def postprocess_chat(stream: AsyncStream[ChatCompletionChunk], stop_words=None, n=1):
    string_buffer = defaultdict(lambda: "")  # key为 choices 数组元素中的index，value为字符串缓冲区
    choices_stop = defaultdict(lambda: False)
    async for data in stream:
        if all(choices_stop.get(i, False) is True for i in range(n)):
            # 中断这个流式
            logger.info(f"all choices stop")
            return
        choice = data.choices[0]
        index = choice.index
        chunk = choice.delta.content
        if chunk is None:
            yield data
            continue
        if choices_stop[index]:
            continue
        remain, is_stop, stop_word = remove_suffix(string_buffer[index], chunk, stop_words)
        choice.delta.content = remain
        if is_stop:
            choices_stop[index] = True
            choice.finish_reason = 'stop'
            logger.info(f"id {data.id}, index {index}, hit stop word: {stop_word.encode('unicode_escape').decode()}")

        string_buffer[index] += chunk
        yield data