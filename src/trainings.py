import copy
import time

import torch
import torch.nn as nn
import torch.optim as optim

from torch.amp import autocast, GradScaler

from .losses import (
    BCEDiceLoss,
    TverskyLoss,
    BCETverskyLoss,
)

def train_model(model, name, train_loader, val_loader, epochs=50, lr=1e-4, patience=5, loss_type = 'BCE', opt = 'Adam', weight_decay = 0.01, alpha = 0.3, beta = 0.7, lambda_t = 0.7, bce_weight = 0.2, smooth = 1):
        
    model.to(device)
    scaler = GradScaler()
    # BCE + Tversky (= Dice with variable FP/FN penalty) loss criterion
    if loss_type == 'BCETversky':
        # Compute pos_weight
        total_pos = 0
        total_neg = 0
        for _, mask in train_dataset:
            mask = mask.float()
            total_pos += mask.sum().item()
            total_neg += mask.numel() - mask.sum().item()
        pos_weight = total_neg / (total_pos + 1e-8)
        print(f'Pos weight: {pos_weight:.4f}')
        pos_weight = torch.tensor([pos_weight], device="cuda")
        criterion = BCETverskyLoss(pos_weight=pos_weight, alpha=alpha, beta=beta, lambda_t=lambda_t)

    # Classical BCE loss 
    if loss_type == 'BCE':
        criterion = nn.BCEWithLogitsLoss()

    # BCE + Dice loss (continuous dice per sample)
    if loss_type == 'BCEDice':
        criterion = BCEDiceLoss(bce_weight=bce_weight, dice_weight=(1-bce_weight), smooth = smooth)

    # Optimizer
    if opt == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    if opt == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    best_val_loss = float('inf')
    best_model_wts = None
    epochs_no_improve = 0
    
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': [], 'train_dice': [], 'val_dice': [], 'train_recall05': [], 'val_recall05':[], 'train_recall07': [], 'val_recall07':[]}
    
    for epoch in range(epochs):
        model.train()
        train_losses, train_accs, train_dices, train_dices_batch, train_recall05, train_recall07 = [], [], [], [], [], []

        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device).float()  # shape [B,1,H,W]
            
            optimizer.zero_grad()

            # Autocast = automate lowering of precision to speed up training when possible 
            with autocast(device_type="cuda"):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            probs = torch.sigmoid(outputs)
            
            train_losses.append(loss.item())

            # Binary predictions 0.7
            preds_bin = (probs > 0.7).float()

            # Verify cancer prediction is not zero, monitor cancer prediction: Recall 
            labels_pos = (labels == 1)
            tp = ((preds_bin == 1) & labels_pos).sum().float()
            fn = ((preds_bin == 0) & labels_pos).sum().float()
            pos_acc = tp / (tp + fn + 1e-12)
            pos_acc = pos_acc.item()
            train_recall07.append(pos_acc)
            
            # Binary predictions
            preds_bin = (probs > 0.5).float()

            # Verify cancer prediction is not zero, monitor cancer prediction: Recall 
            labels_pos = (labels == 1)
            tp = ((preds_bin == 1) & labels_pos).sum().float()
            fn = ((preds_bin == 0) & labels_pos).sum().float()
            pos_acc = tp / (tp + fn + 1e-12)
            pos_acc = pos_acc.item()
            train_recall05.append(pos_acc)
            
            # Accuracy
            acc = (preds_bin == labels).float().mean().item()
            train_accs.append(acc)
            
            # Continuous Dice
            # Per sample + mean over batch
            dice_cont = dice_coefficient_continuous(labels, probs).item()
            train_dices.append(dice_cont)
            # Per batch
            #dice_batch = dice_coefficient(labels, probs).item()
            #train_dices_batch.append(dice_batch)
        
        # Validation
        model.eval()
        val_losses, val_accs, val_dices, val_dices_batch, val_recall07, val_recall05 = [], [], [], [], [], []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device).float()
                outputs = model(inputs)
                probs = torch.sigmoid(outputs)
                val_losses.append(criterion(outputs, labels).item())
                preds_bin = (probs > 0.7).float()
                # Recall 
                labels_pos = (labels == 1)
                tp = ((preds_bin == 1) & labels_pos).sum().float()
                fn = ((preds_bin == 0) & labels_pos).sum().float()
                pos_acc = tp / (tp + fn + 1e-12)
                pos_acc = pos_acc.item()
                val_recall07.append(pos_acc)
                #
                preds_bin = (probs > 0.5).float()
                # Recall 
                labels_pos = (labels == 1)
                tp = ((preds_bin == 1) & labels_pos).sum().float()
                fn = ((preds_bin == 0) & labels_pos).sum().float()
                pos_acc = tp / (tp + fn + 1e-12)
                pos_acc = pos_acc.item()
                val_recall05.append(pos_acc)
                #
                val_accs.append((preds_bin == labels).float().mean().item())
                val_dices.append(dice_coefficient_continuous(labels, probs).item())
                #val_dices_batch.append(dice_coefficient(labels, probs).item())
            
        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses))
        train_acc = float(np.mean(train_accs))
        val_acc = float(np.mean(val_accs))
        train_dice = float(np.mean(train_dices))
        val_dice = float(np.mean(val_dices))
        #train_dice_batch = float(np.mean(train_dices_batch))
        #val_dice_batch = float(np.mean(val_dices_batch))
        train_recall05 = float(np.mean(train_recall05))
        val_recall05 = float(np.mean(val_recall05))
        train_recall07 = float(np.mean(train_recall07))
        val_recall07 = float(np.mean(val_recall07))
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['train_dice'].append(train_dice)
        history['val_dice'].append(val_dice)
        history['train_recall05'].append(train_recall05)
        history['val_recall05'].append(val_recall05)
        history['train_recall07'].append(train_recall07)
        history['val_recall07'].append(val_recall07)
        
        print(f"Epoch {epoch+1}/{epochs} - "
              f"Train loss: {train_loss:.4f}, val loss: {val_loss:.4f}, "
              f"Train acc: {train_acc:.4f}, val acc: {val_acc:.4f}, "
              f"Train dice per sample: {train_dice:.4f}, val dice: {val_dice:.4f}",
              f"Train Recall 0.5: {train_recall05:.4f}, val recall 0.5: {val_recall05:.4f}",
              f"Train Recall 0.7: {train_recall07:.4f}, val recall 0.7: {val_recall07:.4f}")
              #f"Train dice per batch: {train_dice_batch:.4f}, val dice: {val_dice_batch:.4f}")
        
        gn = grad_norms_stats(model)
        print(f"Grad norms — count:{gn['count']}, mean:{gn['mean']:.6f}, std:{gn['std']:.6f}, "
            f"min:{gn['min']:.6f}, max:{gn['max']:.6f}")

        # early stopping
        if (val_recall05 == 0) or (train_recall05 == 0):
            print("Model collapsed, all background predictions.")
            break
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_wts = model.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("Early stopping triggered")
                break
    
    if best_model_wts is not None:
        model.load_state_dict(best_model_wts)

    ###############
    # Save trained model 
    save_path = f"{name}_best.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Model successfully saved to {save_path}")
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {num_params:,}")
    print(f"Trainable parameters: {num_trainable:,}")
    print(f"Model size (FP32): ~{num_params * 4 / 1024**2:.2f} MB")

    del optimizer
    del scaler
    return model, history
