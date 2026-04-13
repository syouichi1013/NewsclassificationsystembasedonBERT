import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer, BertConfig
import os


class Config(object):


    def __init__(self, dataset):
        self.model_name = 'bert'
        self.train_path = dataset + '/data/train.txt'                                # 训练集
        self.dev_path = dataset + '/data/dev.txt'                                    # 验证集
        self.test_path = dataset + '/data/test.txt'                                  # 测试集
        self.class_list = [x.strip() for x in open(
            dataset + '/data/class.txt').readlines()]                                # 类别名单
        self.save_path = dataset + '/saved_dict/' + self.model_name + '.ckpt'        # 模型训练结果
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 设备
        print(f"当前使用设备：{self.device}")
        self.num_classes = len(self.class_list)                         # 类别数
        self.num_epochs = 3                                             # epoch数
        self.batch_size = 16                                          # mini-batch大小
        self.pad_size = 32                                              # 每句话处理成的长度(短填长切)
        self.learning_rate = 5e-5                                       # 学习率
        self.bert_path = './bert_pretrain'
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        self.hidden_size = 768


class Model(nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()
        config_file_path = os.path.join(config.bert_path, 'config.json')
        bert_config = BertConfig.from_json_file(config_file_path)
        bert_config.output_attentions = True
        print(f"检测到词表大小: {bert_config.vocab_size}")
        self.bert = BertModel.from_pretrained(
            config.bert_path,
            config=bert_config,  # 核心修复：把配置传进去
            ignore_mismatched_sizes=True
        )
        for param in self.bert.parameters():
            param.requires_grad = True
        self.fc = nn.Linear(config.hidden_size, config.num_classes)

    def forward(self, x):
        context = x[0]  # 输入的句子
        mask = x[2]  # 对padding部分进行mask，和句子一个size，padding部分用0表示，如：[1, 1, 1, 1, 0, 0]
        outputs = self.bert(context, attention_mask=mask)
        pooled = outputs[1]
        out = self.fc(pooled)
        last_attention = outputs.attentions[-1]
        avg_attention = torch.mean(last_attention, dim=1)
        cls_attention = avg_attention[:, 0, :]
        return out, cls_attention