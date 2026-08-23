from transformers import AutoModel
import torch

from src.utils import FocalLoss

class CustomBertForClassificationAndSpans(torch.nn.Module):
    def __init__(self, model_name="Hate-speech-CNERG/deoffxlmr-mono-tamil"): #l3cube-pune/marathi-roberta
        super(CustomBertForClassificationAndSpans, self).__init__()
        from transformers import AutoModelForTokenClassification, BertModel

        self.bert = AutoModel.from_pretrained(model_name)
        for param in self.bert.parameters():
            param.requires_grad = False
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(self.bert.config.hidden_size, 1024),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(1024, 3)
          )  # Three classes for classification
        self.span_classifier = torch.nn.Sequential(
        torch.nn.Linear(self.bert.config.hidden_size, 1024),  # Span detection head
        torch.nn.ReLU(),
        torch.nn.Dropout(0.1),
        torch.nn.Linear(1024, 1)
        )
    def forward(self, input_ids, attention_mask, labels=None, span_mask=None):
        # Get BERT embeddings
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        # Classification
        logits = self.classifier(outputs.last_hidden_state[:, 0, :])  # Get the [CLS] token embedding for classification

        # Calculate classification loss if labels are provided
        loss = None
        if labels is not None:
            # loss_fct = torch.nn.CrossEntropyLoss()
            loss_fct = FocalLoss()
            loss = loss_fct(logits, labels)

        # Span detection
        span_logits = self.span_classifier(outputs.last_hidden_state)

        # Calculate span loss if span_mask and labels are provided
        if span_mask is not None and labels is not None:
            for label_idx, label in enumerate(labels):
                if label != 0:
                    span_loss = torch.nn.BCEWithLogitsLoss()(span_logits[label_idx].squeeze(-1), span_mask[label_idx].float())
                    loss = loss + span_loss if loss is not None else span_loss

        return {
            "loss": loss,
            "logits": logits,
            "span_logits": span_logits
        }
