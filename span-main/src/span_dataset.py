import torch

class MyDataset(Dataset):
    def __init__(self, texts, classes, spans, tokenizer, max_length=512):
        self.encodings = tokenizer(texts, truncation=True, padding="max_length", max_length=max_length)
        self.labels = classes
        self.spans = spans

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)

        # Create span mask if class is not neutral (class 0 in this case)
        span_mask = [0] * len(item['input_ids'])
        if self.labels[idx] != 0:  # If not the neutral class
            for position in self.spans[idx]:
                if position < len(span_mask):  # Ensure within bounds
                    span_mask[position] = 1
        item['span_mask'] = torch.tensor(span_mask, dtype=torch.long)

        return item

    def __len__(self):
        return len(self.labels)