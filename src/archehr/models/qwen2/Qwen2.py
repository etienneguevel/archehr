from collections import OrderedDict
import torch.nn as nn
from transformers import Qwen2Model


class Qwen2EmbClassification(Qwen2Model):
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path,
        num_labels,
        peft_config,
        *args,
        **kwargs
    ):
        model = super().from_pretrained(
            pretrained_model_name_or_path, *args, **kwargs
        )
        model.add_adapter(peft_config)

        model.embedding_size = model.config.hidden_size
        model.classification_head = model._build_classification_head(
            model.embedding_size,
            num_labels,
        )

        return model

    def _build_classification_head(self, embedding_size, num_labels):
        head = nn.Sequential(OrderedDict([
            ("dropout", nn.Dropout(0.1)),
            ("projection", nn.Linear(embedding_size, num_labels))
        ]))

        return head

    def forward(self,
        input_ids=None,
        inputs_embeds=None,
        attention_mask=None,
        labels=None,
        **kwargs
    ):
        # Ensure only one input type
        if input_ids is not None and inputs_embeds is not None:
            inputs_embeds = None

        outputs = super().forward(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            **kwargs
        )

        pooled_outputs = outputs.last_hidden_state[:, 0]
        logits = self.classification_head(pooled_outputs)

        loss = None
        if labels is not None:
            labels = labels.view(-1)
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)  # assuming labels are shape [B, 1]

        return {"loss": loss, "logits": logits}
        


# class Qwen2Pipeline(Qwen2ForSequenceClassification):

#     def to_layers(self):

#         layers = [
#             *self.model.layers,
#             lambda x: getattr(x, "last_hidden_state")[:, 0],
#             self.dropout,
#             self.classifier
#         ]

#         return layers

#     def forward(self, inputs):

#         inputs_ids, attention_mask = inputs
#         outputs = super().forward(inputs_ids, attention_mask)

#         return outputs, attention_mask
