import torch
import torch.nn as nn
import torch.nn.functional as F

# Dice coefficient and loss
def dice_coefficient(y_true, y_pred, smooth=1): #smooth=1e-6):
    y_true_f = y_true.view(-1)
    y_pred_f = y_pred.view(-1)
    intersection = (y_true_f * y_pred_f).sum()
    return (2. * intersection + smooth) / (y_true_f.sum() + y_pred_f.sum() + smooth)

def dice_coefficient_continuous(y_true, y_pred, smooth=1):
    """Dice using continuous probabilities (y_pred between 0 and 1).
    Per image and then mean, not for the whole batch as a single image."""
    y_true_f = y_true#.view(-1)
    y_pred_f = y_pred#.view(-1)
    intersection = (y_true_f * y_pred_f).sum(dim=(1,2,3))
    dice = (2. * intersection + smooth) / (y_true_f.sum(dim=(1,2,3)) + y_pred_f.sum(dim=(1,2,3)) + smooth)
    return dice.mean()

def dice_coefficient_dilated(y_true, y_pred, radius = 1, smooth=1):
    """Dilated Dice coefficient"""
    y_true = y_true.float()
    y_pred = y_pred.float()
    k = 2 * radius + 1
    pad = radius

    A_d = F.max_pool2d(y_true, kernel_size=k, stride=1, padding=pad)
    P_d = F.max_pool2d(y_pred, kernel_size=k, stride=1, padding=pad)
    inter = (A_d * y_pred).sum(dim=(1,2,3))
    denom = A_d.sum(dim=(1,2,3)) + y_pred.sum(dim=(1,2,3))
    dice1 = (2.0 * inter + smooth) / (denom + smooth)
    inter = (y_true * P_d).sum(dim=(1,2,3))
    denom = y_true.sum(dim=(1,2,3)) + P_d.sum(dim=(1,2,3))
    dice2 = (2.0 * inter + smooth) / (denom + smooth)
    dice_imgs = 0.5 * (dice1 + dice2)
    
    return dice_imgs.mean()

def dice_loss(y_true, y_pred):
    return 1 - dice_coefficient(y_true, y_pred)

# Custom Loss function combining Dice and BCE 
class BCEDiceLoss(nn.Module):
    """
    Combined Binary Cross Entropy + Dice Loss for binary segmentation.
    """
    def __init__(self, smooth=1, bce_weight=0.5, dice_weight=0.5):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()  # keeps logits input for numerical stability
        self.smooth = smooth
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, logits, targets): 
        # BCE part
        bce_loss = self.bce(logits, targets)

        # Dice part
        probs = torch.sigmoid(logits)
        
        targets = targets.float()
        intersection = (probs * targets).sum(dim=(1,2,3))
        dice_score = (2. * intersection + self.smooth) / (
            probs.sum(dim=(1,2,3)) + targets.sum(dim=(1,2,3)) + self.smooth
        )
        dice_loss = 1 - dice_score.mean()

        # Combine
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss
    
class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.3, beta=0.7, eps=1e-7):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        targets = targets.float()

        tp = (probs * targets).sum(dim=(1,2,3))
        fp = ((1 - targets) * probs).sum(dim=(1,2,3))
        fn = (targets * (1 - probs)).sum(dim=(1,2,3))

        tversky_index = tp / (tp + self.alpha*fp + self.beta*fn + self.eps)
        return 1 - tversky_index.mean()
    
class BCETverskyLoss(nn.Module):
    def __init__(self, pos_weight=None, alpha=0.3, beta=0.7, lambda_t=0.7):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.tversky = TverskyLoss(alpha=alpha, beta=beta)
        self.lambda_t = lambda_t

    def forward(self, logits, targets):
        loss_bce = self.bce(logits, targets)
        loss_t = self.tversky(logits, targets)
        return (1 - self.lambda_t) * loss_bce + self.lambda_t * loss_t 

def numpy_dice(pred, true, eps=1e-6):
    # pred, true: flattened numpy arrays with 0/1 values
    pred = pred.astype(np.uint8)
    true = true.astype(np.uint8)
    inter = (pred & true).sum()
    return (2.0 * inter + eps) / (pred.sum() + true.sum() + eps)
