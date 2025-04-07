import torch


class QADataset(torch.utils.data.Dataset):
    def __init__(self, data, tokenizer):
        super(QADataset, self).__init__()
        self.data = data
        self.tokenizer = tokenizer
        self.translate_dict = {
            u: k
            for k, u in enumerate(set([i['label'] for i in data]))
        }

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        query, sentence = item['query']

        encoding = self.tokenizer(
            query,
            sentence,
            padding=False,
            truncation=False,
            return_tensors=None
        )
        encoding['labels'] = self.translate_dict[item['label']]
        return encoding
