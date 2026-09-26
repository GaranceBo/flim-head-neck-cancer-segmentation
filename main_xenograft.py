os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch, torchvision 
from torchvision import models
from torchvision.models import ResNet34_Weights, resnet50, ResNet50_Weights
import numpy as np
import random 
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import cv2
import skimage
from skimage import measure, morphology
from sklearn.utils.class_weight import compute_class_weight
from scipy.stats import wasserstein_distance
from scipy.signal import correlate
from skimage.metrics import structural_similarity as ssim
import psutil
import gc
import math
import seaborn as sns
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve, roc_auc_score, precision_recall_curve, average_precision_score
from scipy.ndimage import binary_dilation
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
import segmentation_models_pytorch as smp
import timeit
import time
import logging
from torch.amp import autocast, GradScaler
from src.data_preprocessing import load_npy_dataset, crop_dataset, downsample_dataset, rotate_image_and_mask, rotate_dataset, flip_image_and_mask, flip dataset, 
get_cropped_dataset, get_augmented_dataset, split_dataset, generate_64_dataset, load_datasets, CancerDataset, custom_collate
from src.evaluation import evaluate_model
from src.losses import dice_coefficient, dice_coefficient_continuous, dice_coefficient_dilated, dice_loss, BCEDiceLoss, TverskyLoss, BCETverskyLoss, numpy_dice
from src.losses import train_model
from src.utils import setup_device, configure_torch, set_seed, dataset_similarity, plot_history, plot_reconstruct, logits_hist_stats, grad_norms_stats, find_best_threshold, get_first_conv, analyze_first_layer, channel_ablation
from src.models import SimpleUNet, ResNet34Segmentation32, SimpleViTSeg

# Free memory
gc.collect()
torch.cuda.empty_cache() 

device = setup_device()
configure_torch()

# 1. Load data
dataset_list_tumors = ['dataset_240513', 'dataset_240716', 'dataset_240715', 'dataset_240627', 'dataset_240617', 'dataset_240703']
tumors_dataset = load_npy_datasets(data, dataset_list_tumors)
# 2. Prepare train/validation/test
name_test = ['dataset_240703'] # Choose unseen test tumor
crop = (256,256) # Choose crop size (in pixel, resolution 0.908um/px) 
input_shape = (256,256,3)
res_value = [None] # Choose resolution reduction factor (in pixel square)
train_ds_64, validation_ds_64, test_ds_64 = load_datasets_xenograft(dataset = tumors_dataset, crop = crop, dataset_test_name = name_test, aug_type = 'none', overlap = True)
# 3. Create train/val/test datasets
train_dataset = CancerDatasetXenograft(train_ds_64, include_info=False) 
val_dataset = CancerDatasetXenograft(validation_ds_64, include_info=False)
test_dataset = CancerDatasetXenograft(test_ds_64, include_info=True)
batch_size = 8
# 4. Initiate training loop
trained_histories = []
seeds = [42, 1337, 2023] # Choose seed(s)
for s, seed in enumerate(seeds):
    print(f"===== RUN seed : {seed} =====")
    set_seed(seed)
    g = torch.Generator()
    g.manual_seed(seed)
    # 5. Create PyTorch datasets/loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=g)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_xenograft)
    # 6. Create model(s)
    models_list = [
        (SimpleUNet(), 'Unet dropout'), 
        (ResNet34Segmentation32(pretrained = True), 'ResNet34 pretrained'), 
        (SimpleViTSeg(input_shape=input_shape), 'Transfo2')
    ] 
    # 7. Launch training(s)
    for model_sk, name in models_list:
        start_time = timeit.default_timer()
        print("############################################")
        print(f"Training model {name}")
        print("############################################")
        model, history = train_model(model_sk, ' '.join([name, 'seed', f'{s}', '256all']), device, train_dataset, train_loader, val_loader, epochs=30, lr=1e-4, patience=5, loss_type = 'BCE')
        elapsed = timeit.default_timer() - start_time
        # Print model training execution time 
        print(f'Execution time of {name} : {elapsed} seconds.')
        trained_histories.append((model, history))
        plot_history(history, name)
        # Empty cache to allocate more memory
        torch.cuda.empty_cache()
        gc.collect()
        print("############################################")
        print(f"Testing model {name}")
        print("############################################")
        num_params = sum(p.numel() for p in model.parameters())
        num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total parameters: {num_params:,}")
        print(f"Trainable parameters: {num_trainable:,}")
        print(f"Model size (FP32): ~{num_params * 4 / 1024**2:.2f} MB")
        # 8. Evaluate model
        model.eval()
        # Evaluate optimal threshold on validation data
        thresholds=np.arange(0, 1, 0.1)
        all_probs = []
        all_gts = []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                outputs = model(inputs)            # logits
                probs = torch.sigmoid(outputs).cpu().numpy()  # shape (B,1,H,W)
                all_probs.append(probs)
                all_gts.append(labels.numpy())
        all_probs = np.concatenate(all_probs, axis=0)  # (N,1,H,W)
        all_gts = np.concatenate(all_gts, axis=0)      # (N,1,H,W)
        best_t = 0.5
        best_score = -1
        for t in thresholds:
            preds = (all_probs >= t)
            scores = []
            for p, g in zip(preds, all_gts):
                y_true_flat = g.ravel().flatten()
                y_pred_flat = p.ravel().flatten()
                scores.append(accuracy_score(y_true_flat, y_pred_flat))
            avg_score = float(np.mean(scores))
            if avg_score > best_score:
                best_score = avg_score
                best_t = float(t)
        print(f'Best threshold is : {best_t:.4f}')
        all_probs, all_labels, data_test_mod = [], [], []
        # Evaluate all models on test set
        print(f'Evaluating model {name} 256 on test set. Collecting datas...')
        with torch.no_grad():
            for inputs, labels, info_dict in test_loader:
                inputs = inputs.to(device)
                outputs = model(inputs)
                probs = torch.sigmoid(outputs).cpu()
                preds = (probs > best_t).float()
                all_probs.append(probs)
                all_labels.append(labels.cpu())
        
                for i in range(len(info_dict)):
                    sample_dict = info_dict[i].copy()  # avoid modifying the original dict

                    # Attach per-sample tensors (detach + squeeze to remove batch/channel dims)
                    sample_dict['probs']  = probs[i, 0].numpy().astype(np.float16)
                    sample_dict['preds']  = preds[i, 0].numpy().astype(np.uint8)
                    sample_dict['labels'] = labels[i, 0].cpu().numpy().astype(np.uint8)
                    sample_dict['image'] = inputs[i, 2].cpu().numpy().astype(np.float16)

                    data_test_mod.append(sample_dict)
        probs = torch.cat(all_probs)
        labels = torch.cat(all_labels)

        # === CHANNEL ABLATION TEST ===
        print("\nRunning channel ablation test...")
        model.eval()
        inputs_sample, labels_sample, _ = next(iter(test_loader))
        inputs_sample = inputs_sample.to(device)
        labels_sample = labels_sample.to(device)
        ablation_results = channel_ablation(model, inputs_sample, labels_sample)
        print("Ablation results:")
        for k, v in ablation_results.items():
            print(f"{k}: {v:.6f}")

        print(f'Done! Evaluating and plotting model...')
        # Plot evaluation ROC curves 
        # Complete evaluation non-dilated mask
        evaluate_model(labels, probs, model_name=name, threshold = best_t)
        # Complete evaluation dilated mask dil = 20pixels (cell size)
        evaluate_model(labels, probs, model_name=name, threshold = best_t, dilated_coef=20)
        # Empty cache to allocate more memory
        torch.cuda.empty_cache()
        gc.collect()
        # 9. Reconstruct model prediction
        print("############################################")
        print(f"Reconstructing model {name}")
        print("############################################")
        # Delete all augmented dataset images 
        data_test_original = [j for j in data_test_mod if j['original'] == True]
        # Find all datasets
        datasets = list({d['dataset'] for d in data_test_original})
        l = 512 # Full image pixel size
        c = crop[0] # Cropped images pixel size 
        for ds in datasets:
            # Create blanc grids
            xs, ys = zip(*[(d['coord1'], d['coord2']) for d in data_test_original if d['dataset'] == ds])
            xmin, xmax, ymin, ymax = min(xs), max(xs), min(ys), max(ys)
            grid_preds, grid_probs, grid_labels = np.zeros((l*(xmax-xmin+1),l*(ymax-ymin+1))),\
                                                        np.zeros((l*(xmax-xmin+1),l*(ymax-ymin+1))), \
                                                        np.zeros((l*(xmax-xmin+1),l*(ymax-ymin+1)))
            # Fill grids
            for d in data_test_original: 
                if d['dataset'] == ds: 
                    x0, x1 =  (((d['coord1']-xmin)*l)+c*(d['crop_y'])), ((((d['coord1']-xmin)*l)+c*(d['crop_y']))+c)
                    y0, y1 = (((d['coord2']-ymin)*l)+c*(d['crop_x'])), ((((d['coord2']-ymin)*l)+c*(d['crop_x']))+c)
                    grid_preds[x0:x1, y0:y1] =  d['preds']
                    grid_probs[x0:x1, y0:y1] =  d['probs']
                    grid_labels[x0:x1, y0:y1] =  d['labels']
            # Plot reconstructed lymph nodes 
            plot_reconstruct(grid_probs, grid_preds, grid_labels, ds, ' '.join([name, 'seed', f'{s}', '256all']), margin = False, show = False)
            # Free memory
            del grid_preds, grid_probs, grid_labels, xs, ys
            gc.collect()
            torch.cuda.empty_cache()
