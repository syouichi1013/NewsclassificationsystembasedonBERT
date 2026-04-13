import numpy as np
import torch
import torch.nn as nn #权重偏置损失函数
import torch.nn.functional as F#函数
from sklearn import metrics#计算模型准确率精确率
from torch.optim import AdamW#导⼊专⻔适配BERT模型的优化器，帮模型调整权重偏置
from tqdm import tqdm

def init_network(model, method='xavier', exclude='embedding', seed=123):
    for name,w in model.named_parameters():
        if exclude not in name:
            if len(w.size())<2:
                continue
            if 'weight' in name:
                if method == 'xavier':
                    nn.init.xavier_normal_(w)
                elif method == 'kaiming':
                    nn.init.kaiming_normal_(w)
                else:
                    nn.init.normal_(w)
            elif 'bias' in name:
                nn.init.constant_(w, 0)
            else:
                pass

def train(config, model, train_iter, dev_iter, test_iter):
    model.train()
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params':[p for n,p in param_optimizer if not any(np in n for np in no_decay)],'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(np in n for np in no_decay)], 'weight_decay': 0.00}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=config.learning_rate)
    total_batch = 0
    dev_best_loss = float('inf')
    model.train()
    for epoch in range(config.num_epochs):
        print('Epoch [{}/{}]'.format(epoch + 1, config.num_epochs))
        for i, (trains, labels, seq_len, mask) in enumerate(train_iter):
            optimizer.zero_grad()
            outputs, _ = model((trains, seq_len, mask))
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()
            if total_batch%1000==0:
                true = labels.data.cpu()
                predict = torch.max(outputs.data, 1)[1].cpu()
                train_acc =  metrics.accuracy_score(true, predict)
                dev_acc, dev_loss = evaluate(config, model, dev_iter)
                if dev_loss < dev_best_loss:
                    dev_best_loss = dev_loss
                    torch.save(model.state_dict(), config.save_path)
                    improve = '*'
                else:
                    improve = ''
                msg = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},  Val Loss: {3:>5.2},  Val Acc: {4:>6.2%} {5}'
                print(msg.format(total_batch, loss.item(), train_acc, dev_loss, dev_acc, improve))
                model.train()
            total_batch += 1
        test(config, model, test_iter)

#输出测试过程中的各个指标结果
def evaluate(config, model, data_iter, test_flag=False):
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    with torch.no_grad():
        for texts, labels, seq_len, mask in data_iter:
            outputs , _= model((texts, seq_len, mask))
            loss = F.cross_entropy(outputs, labels)
            loss_total += loss
            labels = labels.data.cpu().numpy()
            predict = torch.max(outputs.data, 1)[1].cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predict)

    acc = metrics.accuracy_score(labels_all, predict_all)
    if test_flag:
        report = metrics.classification_report(labels_all, predict_all, target_names=config.class_list, digits=4)
        confusion = metrics.confusion_matrix(labels_all, predict_all)
        return acc, loss_total / len(data_iter), report, confusion
    return acc, loss_total / len(data_iter)

def test(config, model, test_iter):
    # test
    model.load_state_dict(torch.load(config.save_path))
    model.eval()
    test_acc, test_loss, test_report, test_confusion = evaluate(config, model, test_iter, test_flag=True)
    msg = 'Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}'
    print(msg.format(test_loss, test_acc))
    print("Precision, Recall and F1-Score...")
    print(test_report)
    print("Confusion Matrix...")
    print(test_confusion)




#先初始化模型参数再调整，然后训练模型更新参数，得出精确率（需要在cpu上计算）,在测试集上尝试是否损失率下降，是的话保存当下模型。最后test。
#其中转成numpy是因为有些参数只能在cpu上计算，并且有些函数只能用数组不能用张量