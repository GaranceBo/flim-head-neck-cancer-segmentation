from src.models import ResNet34Segmentation32, UNetPPcomplete
from src.data import load_datasets, CancerDataset, custom_collate
from src.training import train_model
from src.evaluation import evaluate_model

from src.data_preprocessing import load_npy_dataset, crop_dataset, downsample_dataset, rotate_image_and_mask, rotate_dataset, flip_image_and_mask, flip dataset, 
get_cropped_dataset, get_augmented_dataset, split_dataset, generate_64_dataset, load_datasets, CancerDataset, custom_collate

# 1. Load data

# 2. Prepare train/validation/test

# 3. Create PyTorch datasets/loaders

# 4. Create model
model = ResNet34Segmentation32(pretrained=True)

# 5. Train
model, history = train_model()

# 6. Evaluate
