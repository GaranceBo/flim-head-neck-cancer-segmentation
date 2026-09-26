import gc
import math
import random
import time
import numpy as np
import matplotlib.pyplot as plt
import torch

def setup_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        print(f"cuDNN version: {torch.backends.cudnn.version()}")

    return device

def configure_torch():
    if torch.cuda.is_available():
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(False)
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True

    torch.set_float32_matmul_precision("high")

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def dataset_similarity(dataset, dataset_test=None, groups=None, bins=100):

    # Merge if dataset is a list of datasets
    if (isinstance(dataset, list) and len(dataset) > 0 and isinstance(dataset[0], list)):
        print("Merging multiple datasets...")
        dataset = [item for subdataset in dataset for item in subdataset]

    # Merge datasets
    if dataset_test is not None:
        print("Merging dataset and dataset_test...")
        dataset = list(dataset) + list(dataset_test)

    # Random shuffle
    np.random.shuffle(dataset)

    for a in range(len(dataset)):
        vmin1,vmax1=np.percentile(dataset[a]['image'][0],(1,99))
        vmin2,vmax2=np.percentile(dataset[a]['image'][1],(1,99))
        vmin3,vmax3=np.percentile(dataset[a]['image'][2],(1,99))
        dataset[a]['image'][0]=np.clip(dataset[a]['image'][0],vmin1,vmax1)
        dataset[a]['image'][1]=np.clip(dataset[a]['image'][1],vmin2,vmax2)
        dataset[a]['image'][2]=np.clip(dataset[a]['image'][2],vmin3,vmax3)
        dataset[a]['image'][0]/=10
        dataset[a]['image'][1]/=10

    # Normalize group formatting
    normalized=[]
    for g in groups:
        if isinstance(g,str):
            normalized.append([g])
        else:
            normalized.append(g)
    groups=normalized

    # Extract images from group
    def collect(group):
        imgs=[]
        for d in dataset:
            if d['dataset'] in group:
                imgs.append(d['image'])
        return imgs
    
    # Build average histogram
    def average_histograms(images):
        hists=[]
        for img in images:
            channel_hists=[]
            for c in range(3):
                h,_=np.histogram(img[c].flatten(),bins=bins,density=True)
                h=h/(h.sum()+1e-12)
                channel_hists.append(h)
            hists.append(channel_hists)
        hists=np.array(hists)
        return hists.mean(axis=0)
    
    # Mean image
    def average_image(images):
        return np.mean(images,axis=0)

    # Compare groups
    def compare(group1,group2):
        imgs1=collect(group1)
        imgs2=collect(group2)
        hist1=average_histograms(imgs1)
        hist2=average_histograms(imgs2)
        mean1=average_image(imgs1)
        mean2=average_image(imgs2)
        wasserstein_scores=[]
        correlation_scores=[]
        ssim_scores=[]

        for c in range(3):
            # Wasserstein
            w=wasserstein_distance(np.arange(bins),np.arange(bins),hist1[c],hist2[c])
            wasserstein_scores.append(w)

            # histogram correlation
            corr=np.corrcoef(hist1[c],hist2[c])[0,1]
            correlation_scores.append(corr)

            # SSIM
            s=ssim(mean1[c],mean2[c],data_range=mean2[c].max()-mean2[c].min())
            ssim_scores.append(s)

        print("\n"+"="*60)
        print(f"Similarity between "f"{group1}"f" and "f"{group2}")
        print("\nWasserstein:")
        for c,w in enumerate(wasserstein_scores):
            print(f"Channel {c}: {w:.4f}")
        print(f"Mean: "f"{np.mean(wasserstein_scores):.4f}")
        print("\nHistogram correlation:")
        for c,w in enumerate(correlation_scores):
            print(f"Channel {c}: {w:.4f}")
        print(f"Mean: "f"{np.mean(correlation_scores):.4f}")
        print("\nSSIM:")
        for c,w in enumerate(ssim_scores):
            print(f"Channel {c}: {w:.4f}")
        print(f"Mean: "f"{np.mean(ssim_scores):.4f}")

    # pairwise
    for i in range(len(groups)):
        current=groups[i]
        for j in range(i+1,len(groups)):
            compare(current,groups[j])
        remaining=[]
        for j,g in enumerate(groups):
            if i!=j:
                remaining.extend(g)
        compare(current,remaining) # Dataset class for PyTorch
    
# Plotting function for training history
def plot_history(history, model_name):
    plt.figure(figsize=(24, 5))
    
    # Accuracy
    plt.subplot(1, 4, 1)
    plt.plot(history['train_acc'], label='Training Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title(f'Accuracy - {model_name}')
    
    # Loss
    plt.subplot(1, 4, 2)
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title(f'Loss - {model_name}')
    
    # Dice Coefficient
    plt.subplot(1, 4, 3)
    plt.plot(history['train_dice'], label='Training Dice Coefficient')
    plt.plot(history['val_dice'], label='Validation Dice Coefficient')
    plt.xlabel('Epoch')
    plt.ylabel('Dice Coefficient')
    plt.legend()
    plt.title(f'Dice Coefficient - {model_name}')

    # Recall Threshold 0.5
    plt.subplot(1, 4, 4)
    plt.plot(history['train_recall05'], label='Training Recall (th=0.5)', color='tab:blue')
    plt.plot(history['val_recall05'], label='Validation Recall (th=0.5)', color='tab:orange')
    plt.plot(history['train_recall07'], label='Training Recall (th=0.7)', linestyle='--', color='tab:blue')
    plt.plot(history['val_recall07'], label='Validation Recall (th=0.7)', linestyle='--', color='tab:orange')
    plt.xlabel('Epoch')
    plt.ylabel('Recall')
    plt.legend()
    plt.title(f'Recall - {model_name}')
    
    plt.tight_layout()
    plt.show()

# Reconstruct total grid function
# Plot function
def plot_reconstruct(grid_probs, grid_preds, grid_labels, dataset_name, model_name,
                     alpha_pred=0.4, alpha_prob=0.5, show=False, save_path=None, margin = False, grid_image = None):
    """
    Display or save grids.

    grid_probs: float32 array in [0,1]
    grid_preds: binary array {0,1}
    grid_labels: binary array {0,1}
    """

    # --- Ensure arrays are float32 and same shape ---
    #cleaned = cleaned.astype(np.float32)
    grid_probs = grid_probs.astype(np.float32)
    grid_preds = grid_preds.astype(np.float32)
    grid_labels = grid_labels.astype(np.float32)
    if grid_image is not None: 
        grid_image = grid_image.astype(np.float32)

    # --- Create base grayscale mask for visualization ---
    #cl = cleaned
    gt = grid_labels
    pred_bin = grid_preds
    prob = grid_probs
    if grid_image is not None:
        imag = grid_image

    # --- Create overlays ---
    # Red overlay (prediction on ground truth)
    base_rgb = np.stack([gt, gt, gt], axis=-1)
    red_overlay = np.zeros_like(base_rgb)
    red_overlay[..., 0] = pred_bin
    binary_overlay = np.clip(base_rgb * (1 - alpha_pred) + red_overlay * alpha_pred, 0, 1)

    # --- Probabilistic overlay (jet colormap) ---
    prob_rgb = plt.cm.viridis(prob)[:, :, :3]  # remove alpha
    prob_overlay = np.clip(base_rgb * (1 - alpha_prob) + prob_rgb * alpha_prob, 0, 1)

    # --- Matplotlib figure ---
    height, width = grid_labels.shape
    dpi = 600  
    figsize = (width / dpi, height / dpi)  # in inches
    if grid_image is not None:
        fig, axes = plt.subplots(1, 4, figsize=(figsize[0]*4, figsize[1]), dpi=dpi)
    else: 
        fig, axes = plt.subplots(1, 3, figsize=(figsize[0]*3, figsize[1]), dpi=dpi)
    #fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    plt.subplots_adjust(wspace=0.05, hspace=0)

    # (a) Ground Truth
    cmap_bin = plt.cm.gray.copy()
    cmap_bin.set_under('gray')
    axes[0].imshow(gt,cmap=cmap_bin,vmin=0,vmax=1,interpolation='nearest')
    axes[0].set_title(f'{dataset_name} - Ground Truth', fontsize=40)
    axes[0].axis('off')

    # (b) Predicted Binary (before postprocessing)
    cmap_bin = plt.cm.gray.copy()
    cmap_bin.set_under('gray')
    axes[1].imshow(pred_bin,cmap=cmap_bin,vmin=0,vmax=1,interpolation='nearest')
    axes[1].set_title('Predicted Binary Mask', fontsize=40)
    axes[1].axis('off')

    # (b) Predicted Binary (after postprocessing)
    #axes[2].imshow(cl, cmap='gray', vmin=0, vmax=1, interpolation='nearest')
    #axes[2].set_title('Predicted Binary Mask (Post)', fontsize=40)
    #axes[2].axis('off')

    # (c) Red overlay
    #axes[2].imshow(binary_overlay, interpolation='nearest')
    #axes[2].set_title('Binary Prediction Overlay', fontsize=40)
    #axes[2].axis('off')

    # (d) Probabilistic (no overlay)
    cmap_prob = plt.cm.viridis.copy()
    cmap_prob.set_under('gray')
    valid_probs = prob[prob >= 0]
    im = axes[2].imshow(prob,cmap=cmap_prob,vmin=0,vmax=np.max(valid_probs),interpolation='nearest')
    axes[2].set_title('Probabilistic Map', fontsize=40)
    axes[2].axis('off')

    # (b) Predicted Binary (before postprocessing)
    if grid_image is not None:
        cmap_img = plt.cm.inferno.copy()
        cmap_img.set_under('gray')
        valid_img = imag[imag >= 0]
        axes[3].imshow(imag, cmap=cmap_img, vmin=np.min(valid_img), vmax=np.max(valid_img), interpolation='nearest')
        axes[3].set_title('Input tissue (intensity)', fontsize=40)
        axes[3].axis('off')

    # Add colorbar for probabilistic map
    cbar = fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
    cbar.set_label('Probability', fontsize=30)
    cbar.ax.tick_params(labelsize=30)

    # --- Save in high resolution ---
    if margin: 
        output_path = save_path or f"Reconstruction_{model_name}_{dataset_name}_margin.png"
    else: 
        output_path = save_path or f"Reconstruction_{model_name}_{dataset_name}.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=600, bbox_inches='tight')  # high-quality save
    if show:
        plt.show()
    else:
        plt.close(fig)

    print(f"Saved reconstruction figure to {output_path}")

def logits_hist_stats(logits, nbins=50):
    """Return histogram counts and basic stats for logits tensor (CPU)."""
    flat = logits.detach().cpu().view(-1).numpy()
    counts, bin_edges = np.histogram(flat, bins=nbins, range=(float(flat.min()), float(flat.max())))
    stats = {
        'min': float(flat.min()),
        'max': float(flat.max()),
        'mean': float(flat.mean()),
        'std': float(flat.std()),
        'median': float(np.median(flat)),
        'counts': counts,
        'bin_edges': bin_edges
    }
    return stats

def grad_norms_stats(model):
    norms = []
    named = []
    for name, p in model.named_parameters():
        if p.grad is not None:
            n = p.grad.detach().cpu().norm().item()
            norms.append(n)
            named.append((name, n))
    if len(norms) == 0:
        return {'count': 0}
    arr = np.array(norms)
    stats = {
        'count': len(norms),
        'mean': float(arr.mean()),
        'std': float(arr.std()),
        'min': float(arr.min()),
        'max': float(arr.max()),
        'per_param': named
    }
    return stats

def find_best_threshold(model, val_loader, device, thresholds=np.arange(0, 1, 0.1)):
    model.eval()
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
    # compute per-image dice averaged
    for t in thresholds:
        preds = (all_probs >= t).astype(np.uint8)
        scores = []
        for p, g in zip(preds, all_gts):
            scores.append(numpy_dice(p.ravel(), g.ravel()))
        avg_score = float(np.mean(scores))
        if avg_score > best_score:
            best_score = avg_score
            best_t = float(t)
    print(f'Best threshold is : {best_t}')
    return best_t ### CHANNEL USAGE

def get_first_conv(module):
    for m in module.modules():
        if isinstance(m, torch.nn.Conv2d):
            return m
    raise ValueError("No Conv2d layer found in model")

def analyze_first_layer(model):
    conv1 = get_first_conv(model)

    weights = conv1.weight.data
    importance = weights.abs().mean(dim=(0, 2, 3))

    print("\nChannel importance:")
    for i, val in enumerate(importance):
        print(f"Channel {i}: {val.item():.6f}")

def channel_ablation(model, x, y):
    results = {}
    with torch.no_grad():
        base_out = model(x)
        base_loss = ((torch.sigmoid(base_out) - y)**2).mean().item()
        results["full"] = base_loss
        for ch in range(x.shape[1]):
            x_mod = x.clone()
            x_mod[:, ch, :, :] = 0  # remove channel
            out = model(x_mod)
            loss = ((torch.sigmoid(out) - y)**2).mean().item()
            results[f"no_channel_{ch}"] = loss
    return results # Instantiate models
