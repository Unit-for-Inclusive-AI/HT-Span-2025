from sklearn.metrics import confusion_matrix, classification_report,accuracy_score,f1_score,precision_score,recall_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments, AutoTokenizer
import sklearn
from sklearn.model_selection import train_test_split
import ast

from src.models.bert_bilstm_crf import CustomBertForClassificationAndSpans
from src.utils import compute_metrics
from src.span_dataset import MyDataset

df = pd.read_csv("/content/final_tamil.csv")
df = df.rename(columns={'spans': 'label', 'labels': 'class'})
df['label'] = df['label'].fillna('[]')

# df = pd.read_csv("/content/english_span_updated.csv")
# df = pd.read_csv("/content/marathi_combined_updated_data.csv")

# df['label'] = df['label'].apply(lambda x: [int(i) for i in x.split(',')])
df['label'] = df['label'].apply(lambda x: [int(i) for i in x.split(',')] if x != '[]' else [])
df['class'] = df['class'].fillna('None of the above ')
label_mapping = {
                'None of the above ': 0,
                'Transphobia': 1,
                'Homophobia':2
            }
df['class'] = df['class'].map(label_mapping)
df['class'] = df['class'].astype(int)
train_df,val_df = train_test_split(df,test_size=0.15,random_state = 427,stratify = df['class'])
print(train_df.shape, val_df.shape)
# Convert spans from string to list (if stored as a string representation of a list)
train_df['label'] = train_df['label'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

train_texts = train_df['text'].tolist()
train_classes = train_df['class'].tolist()
train_spans = train_df['label'].tolist()

# Convert spans from string to list (if stored as a string representation of a list)
val_df['label'] = val_df['label'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

val_texts = val_df['text'].tolist()
val_classes = val_df['class'].tolist()
val_spans = val_df['label'].tolist()

model = CustomBertForClassificationAndSpans()
tokenizer = AutoTokenizer.from_pretrained("Hate-speech-CNERG/deoffxlmr-mono-tamil") #l3cube-pune/marathi-roberta
train_dataset = MyDataset(train_texts, train_classes, train_spans, tokenizer)
eval_dataset = MyDataset(val_texts, val_classes, val_spans, tokenizer)

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    logging_dir='./logs',
    metric_for_best_model="f1",        
    greater_is_better=True,
    max_grad_norm=1.0,
    learning_rate=4e-3
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics
)

trainer.train()

predictions = trainer.predict(eval_dataset)
predicted_labels = predictions.predictions[0].argmax(-1)
true_labels = predictions.label_ids

label_mapping = {
                'None': 0,
                'Transphobia': 1,
                'Homophobia':2
            }

cm = confusion_matrix(true_labels, predicted_labels)
print(cm)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=label_mapping.keys(), yticklabels=label_mapping.keys())
plt.title("Confusion Matrix for Hate-RoBERTa-Tamil + Bi-LSTM + CRF")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

report = classification_report(true_labels, predicted_labels, target_names=label_mapping.keys())
print(report)

print(f'Accuracy: {accuracy_score(true_labels, predicted_labels)}')
print(f'F1 Score: {f1_score(true_labels, predicted_labels, average="macro")}')
print(f'Precision: {precision_score(true_labels, predicted_labels, average="macro")}')
print(f'Recall: {recall_score(true_labels, predicted_labels, average="macro")}')