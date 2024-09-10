# chatgpt-server-python
使用 [fastapi](https://github.com/fastapi/fastapi) 和 [openai-python](https://github.com/openai/openai-python) 实现的 chatgpt-server，适配了openai 的输入输出格式，支持流式调用和非流式调用。流式调用支持context cancel。

openai_api_protocol.py 来源于 [openai_api_protocol.py](https://github.com/lm-sys/FastChat/blob/main/fastchat/protocol/openai_api_protocol.py)

## 使用指南
安装依赖
```bash
pip install -r requirements.txt
```

启动服务
```bash
python main.py
```

使用下面命令测试
```bash
curl localhost:8080/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
        "model": "codewise-7b",
        "prompt": "如何使用nginx进行负载均衡?",
        "max_tokens": 256,
        "temperature": 0.2,
        "stream": true
    }'

curl localhost:8080/v1/chat/completions \
  -X POST \
  -H "Content-Type: application/json" \
  -d '{
    "model": "codewise-7b",
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "如何使用nginx进行负载均衡？"}
    ],
    "max_tokens": 256,
    "temperature": 1,
    "stream": true,
    "skip_special_tokens": false
  }'
```