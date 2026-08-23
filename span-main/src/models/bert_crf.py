from transformers import AutoModel
import torch
from torchcrf import CRF
from src.utils import FocalLoss

class CustomBertForClassificationAndSpans(torch.nn.Module):
    def __init__(self, model_name="Hate-speech-CNERG/deoffxlmr-mono-tamil"):
        super(CustomBertForClassificationAndSpans, self).__init__()
        from transformers import AutoModel
        self.bert = AutoModel.from_pretrained(
        model_name,
        output_hidden_states=True,
        return_dict=True
    )
        for param in self.bert.parameters():
            param.requires_grad = False
        self.classifier = torch.nn.Linear(self.bert.config.hidden_size, 3)  # Three classes for classification
        self.span_classifier = torch.nn.Linear(self.bert.config.hidden_size, 1)  # Span detection head
        self.crf = CRF(num_tags = 3, batch_first = True)  # Assuming 3 tags for span prediction
        self.crf_inp = torch.nn.Linear(self.bert.config.hidden_size, 3)
    def forward(self, input_ids, attention_mask, labels=None, span_mask=None):
        # Get BERT embeddings
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        pooled_output = outputs.pooler_output
        # Classification logits (for [CLS] token)
        logits = self.classifier(pooled_output)
        crf_logits = self.crf_inp(sequence_output)
        # Calculate classification loss (FocalLoss or CrossEntropyLoss)
        classification_loss = None
        if labels is not None:
            loss_fct = FocalLoss()  # Replace with CrossEntropyLoss if desired
            classification_loss = loss_fct(logits, labels)

        # Span detection logits
        span_logits = self.span_classifier(pooled_output)

        # CRF loss for spans
        crf_loss = None
        if span_mask is not None and labels is not None:
            crf_loss = -self.crf(crf_logits, span_mask, mask=attention_mask.byte(),reduction = "mean")

        # Combine losses if both are present
        total_loss = classification_loss + crf_loss if classification_loss is not None and crf_loss is not None else classification_loss or crf_loss

        return {
            "loss": total_loss,
            "logits": logits,
            "span_logits": span_logits
        }
