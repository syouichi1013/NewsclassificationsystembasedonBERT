from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import bert
import os
import numpy as np
import uvicorn

app = FastAPI()

# 允许跨域请求，这样你直接双击打开 HTML 也能访问后端
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. 加载配置和模型
dataset = 'THUCNews'
config = bert.Config(dataset)
config.device = torch.device('cpu')  # 推理建议用 CPU，更稳定

model = bert.Model(config).to(config.device)
if os.path.exists(config.save_path):
    model.load_state_dict(torch.load(config.save_path, map_location='cpu'))
    model.eval()
    print("🚀 BERT Model Loaded Successfully!")
else:
    print(f"❌ 错误: 找不到模型权重文件 {config.save_path}")


class NewsInput(BaseModel):
    content: str


@app.post("/api/analyze")
async def predict_news(item: NewsInput):
    text = item.content
    if not text:
        raise HTTPException(status_code=400, detail="内容不能为空")

    # 2. 文本预处理 (与训练保持一致)
    tokens = config.tokenizer.tokenize(text)
    tokens = ['[CLS]'] + tokens
    seq_len = len(tokens)

    if seq_len > config.pad_size:
        tokens = tokens[:config.pad_size]
        seq_len = config.pad_size

    token_ids = config.tokenizer.convert_tokens_to_ids(tokens)
    mask = [1] * seq_len + [0] * (config.pad_size - seq_len)
    token_ids += [0] * (config.pad_size - seq_len)

    # 3. 转换为 Tensor
    ids = torch.LongTensor([token_ids]).to(config.device)
    mask_tensor = torch.LongTensor([mask]).to(config.device)

    # 4. 模型推理
    with torch.no_grad():
        # 对应你 bert.py 里的 return out, cls_attention
        logits, cls_attention = model((ids, None, mask_tensor))

        # 计算概率
        probs = torch.nn.functional.softmax(logits, dim=1).cpu().numpy()[0]
        pred_idx = np.argmax(probs)
        label = config.class_list[pred_idx]

    # 5. 提取注意力权重 (用于前端颜色深浅展示)
    # 注意：bert.py 中已经做了 mean 处理，这里直接取第 0 条数据的长度部分
    attention_weights = cls_attention[0][:seq_len].tolist()

    return {
        "label": label,
        "probability": float(probs[pred_idx]),
        "all_probs": {config.class_list[i]: float(probs[i]) for i in range(len(probs))},
        "tokens": tokens,
        "attention_weights": attention_weights
    }


if __name__ == "__main__":
    # 统一使用 8000 端口
    uvicorn.run(app, host="127.0.0.1", port=8000)