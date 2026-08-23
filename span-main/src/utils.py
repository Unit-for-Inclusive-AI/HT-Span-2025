from torch import nn
import torch.nn.functional as F
import torch
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

class FocalLoss(nn.Module):
    def __init__(self, class_counts = [149, 146, 306], gamma=2, reduction='mean'):
        """
        Initialize Focal Loss with class weights.
        :param class_counts: List or tensor of counts for each class.
        :param gamma: Focusing parameter, reduces the loss for well-classified examples, putting more focus on hard examples.
        :param reduction: Specifies the reduction to apply to the output ('none', 'mean', or 'sum').
        """
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction

        total_count = sum(class_counts)
        class_weights = [total_count / count for count in class_counts]
        self.class_weights = torch.tensor(class_weights, dtype=torch.float32)

    def forward(self, inputs, targets):

        self.class_weights = self.class_weights.to(inputs.device)

        ce_loss = F.cross_entropy(
            inputs, targets, reduction='none', weight=self.class_weights, label_smoothing=0.25
        )
        pt = torch.exp(-ce_loss)

        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

def compute_metrics(pred):
    preds = pred.predictions[0]
    labels = pred.label_ids

    class_preds = preds.argmax(-1)
    class_precision, class_recall, class_f1, _ = precision_recall_fscore_support(labels, class_preds, average='weighted')
    class_accuracy = accuracy_score(labels, class_preds)

    return {
        'accuracy': class_accuracy,
        'f1': class_f1,
        'precision': class_precision,
        'recall': class_recall,
    }