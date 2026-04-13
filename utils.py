import torch
from tqdm import tqdm


PAD, CLS = '[PAD]', '[CLS]' #cls相当于文本摘要，放在文本开头，bert通过这个符号总结文本核心信息；pad填充符号，为了所有的文本长度对齐

def build_dataset(config):
    def load_dataset(path,pad_size=32):
        contents=[]
        with open(path, 'r',encoding='UTF-8') as f:
            for line in tqdm(f):
                lin=line.strip()
                if not lin:continue
                text,label=lin.split('\t')
                token=config.tokenizer.tokenize(text)
                token=[CLS]+token
                seq_len=len(token)
                mask = []
                token_ids=config.tokenizer.convert_tokens_to_ids(token)
                if seq_len<pad_size:
                    mask=[1]*seq_len+[0]*(pad_size-seq_len)
                    token_ids+=[0]*(pad_size-seq_len)
                else:
                    mask=[1]*pad_size
                    token_ids=token_ids[:pad_size]
                    seq_len=pad_size
                contents.append((token_ids, int(label), seq_len, mask))
        return contents
    train=load_dataset(config.train_path,config.pad_size)
    dev=load_dataset(config.dev_path,config.pad_size)
    test=load_dataset(config.test_path,config.pad_size)
    return train,dev,test

class DatasetIterater(object):
    def __init__(self, batches,batch_size,device):#括号里面只写外面传递进来的
        self.batches=batches
        self.batch_size = batch_size
        self.n_batches=len(batches)//batch_size
        self.residue=False #是否保留最后不足一批的数据
        if len(batches)%batch_size!=0:
            self.residue=True
        self.index=0
        self.device=device

    def _to_tensor(self,datas):
        x = torch.LongTensor(list(_[0] for _ in datas)).to(self.device)
        y = torch.LongTensor(list(_[1] for _ in datas)).to(self.device)
        seq_len = torch.LongTensor(list(_[2] for _ in datas)).to(self.device)
        mask = torch.LongTensor(list(_[3] for _ in datas)).to(self.device)
        return x, y, seq_len, mask

    def __next__(self):
        if self.index< self.n_batches:
            batches=self.batches[self.index*self.batch_size:(self.index+1)*self.batch_size]
            self.index+=1
            batches = self._to_tensor(batches)
            return batches

        elif self.residue and self.index== self.n_batches:
            batches=self.batches[self.index*self.batch_size: len(self.batches)]
            self.index+=1
            batches = self._to_tensor(batches)
            return batches

        else:
            self.index = 0
            raise StopIteration

    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches

def build_iterator(dataset, config):
    iter = DatasetIterater(dataset, config.batch_size, config.device)
    return iter#每次返回的都是__next__里面的返回值


# iter是迭代器 存的是数据和动作
#build_dataset：把文本 → 数字  build_iterator：把数字 → 交给迭代器 DatasetIterater：把数字 → 张量（向量）

