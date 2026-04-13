import torch
import bert
import numpy as np


def run_simple_test():
    # 1. 环境准备
    dataset = 'THUCNews'
    config = bert.Config(dataset)
    model = bert.Model(config).to(config.device)

    # 2. 加载你训练好的那个 .ckpt 文件
    model.load_state_dict(torch.load(config.save_path, map_location=config.device))
    model.eval()

    print("--- 模型已就绪，请输入新闻标题（输入 q 退出） ---")

    while True:
        text = input("\n请输入标题: ")
        if text.lower() == 'q': break

        # 3. 必须的“三步翻译” (把汉字变数字)
        tokens = ['[CLS]'] + config.tokenizer.tokenize(text)
        ids = config.tokenizer.convert_tokens_to_ids(tokens)[:32]  # 只要前32个字
        ids += [0] * (32 - len(ids))  # 不够32个字的补0

        # 4. 送入模型并拿结果
        input_tensor = torch.LongTensor([ids]).to(config.device)
        with torch.no_grad():
            outputs = model((input_tensor, None, input_tensor))  # 对应 forward(self, x)
            logits = outputs[0] if isinstance(outputs, tuple) else outputs
            ans = torch.argmax(logits, dim=1).item()

            print(f">> 分类结果: {config.class_list[ans]}")


if __name__ == '__main__':
    run_simple_test()