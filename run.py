import torch
import numpy as np
from train import train, init_network
from utils import build_dataset, build_iterator
import os
import bert


print("CUDA是否可用：", torch.cuda.is_available())
print("GPU数量：", torch.cuda.device_count())
print("GPU名称：", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "无")

if __name__ == '__main__':
    dataset = 'THUCNews'  # 数据集

    config = bert.Config(dataset)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样

    save_dir = os.path.dirname(config.save_path)  # 提取保存路径的目录（THUCNews/saved_dict）
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"创建文件夹：{save_dir}")


    print("Loading data...")
    print("训练集路径：", config.train_path)
    print("文件是否存在：", os.path.exists(config.train_path))

    train_data, dev_data, test_data = build_dataset(config)
    train_iter = build_iterator(train_data, config)
    dev_iter = build_iterator(dev_data, config)
    test_iter = build_iterator(test_data, config)

    # train
    model = bert.Model(config).to(config.device)
    train(config, model, train_iter, dev_iter, test_iter)

def quick_predict(text):
    pad_size = config.pad_size
    token = config.tokenizer.tokenize(text)
    token = ['[CLS]'] + token
    seq_len = len(token)
    token_ids = config.tokenizer.convert_tokens_to_ids(token)

    if seq_len < pad_size:
        mask = [1] * seq_len + [0] * (pad_size - seq_len)
        token_ids += [0] * (pad_size - seq_len)
    else:
        mask = [1] * pad_size
        token_ids = token_ids[:pad_size]
        seq_len = pad_size

    # 2. 变成 Tensor（这步必须有，模型只吃 Tensor）
    ids = torch.LongTensor([token_ids]).to(config.device)
    mask_tensor = torch.LongTensor([mask]).to(config.device)
    seq_len_tensor = torch.LongTensor([seq_len]).to(config.device)

    # 3. 推理
    with torch.no_grad():
        outputs = model((ids, seq_len_tensor, mask_tensor))
        logits = outputs[0] if isinstance(outputs, tuple) else outputs
        pred = torch.argmax(logits, dim=1).item()
        return config.class_list[pred]


# 直接测试
while True:
    title = input("\n请输入新闻标题(q退出): ")
    if title == 'q': break
    print(f"分类结果 -> {quick_predict(title)}")