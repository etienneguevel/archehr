import torch

from archehr.data.utils import last_token_pool


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


class QADatasetEmbedding(torch.utils.data.Dataset):
    def __init__(self, data, tokenizer, model, device=torch.device('cpu')):
        super(QADatasetEmbedding, self).__init__()
        self.data = data
        self.tokenizer = tokenizer
        self.model = model.to(device)
        self.device = device
        self.translate_dict = {
            u: k
            for k, u in enumerate(set([i['label'] for i in data]))
        }

    @property
    def emb_size(self):
        return self[0][0].size()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        query, sentence = item['query']

        # make the encoding
        encoding = self.tokenizer(
            query,
            sentence,
            padding=False,
            truncation=False,
            return_tensors='pt'
        )
    
        if isinstance(encoding, torch.Tensor):
            encoding = encoding.to(self.device)

        else:
            encoding = {
                k: v.to(self.device)
                for k, v in encoding.items()
            }
            
        # make the embedding
        with torch.no_grad():
            outputs = self.model(**encoding)
            embedding = last_token_pool(
                outputs.last_hidden_state,
                encoding['attention_mask']
            )

        
        return (
            embedding.squeeze(0).to(self.device),
            self.translate_dict[item['label']]
        )
