import torch
from torch import Tensor
from tqdm import tqdm

from archehr.data.utils import last_token_pool, to_device
from archehr.utils.loaders import DeviceType, load_model_hf


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
    def __init__(self, data, model_name, device: DeviceType = "cpu"):
        super(QADatasetEmbedding, self).__init__()
        self.data = data
        self.model_name = model_name
        self._vector_store = self._make_vector_store(model_name, device)
        self.translate_dict = {
            "essential": 0,
            "not-relevant": 1,
            "supplementary": 1,
        }

    @property
    def emb_size(self):
        return self[0][0].size()

    def _make_vector_store(self, model_name, device):

        # Load the model and tokenizer
        model, tokenizer = load_model_hf(model_name)
        model.eval()

        # Get the device
        device = model.device

        vector_store = []

        for item in tqdm(self.data):
            query, sentence = item['query']

            # make the encoding
            encoding = tokenizer(
                query,
                sentence,
                padding=False,
                truncation=False,
                return_tensors='pt'
            )

            # move to the correct device
            encoding = to_device(encoding, model.device)

            # make the embedding
            with torch.no_grad():
                if isinstance(encoding, Tensor):
                    outputs = model(encoding)
                else:
                    outputs = model(**encoding)

                embedding = last_token_pool(
                    outputs.last_hidden_state,
                    encoding['attention_mask']
                )
            
            vector_store.append(embedding.squeeze(0).to(device))

        return torch.stack(vector_store)


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        vector = self._vector_store[idx, :]

        return (
            vector,
            self.translate_dict[item['label']]
        )
