from src.data_preprocessing import load_npy_dataset, crop_dataset, downsample_dataset, rotate_image_and_mask, rotate_dataset, flip_image_and_mask, flip dataset, 
get_cropped_dataset, get_augmented_dataset, split_dataset, generate_64_dataset, load_datasets, CancerDataset, custom_collate
from src.evaluation import evaluate_model
from src.losses import dice_coefficient, dice_coefficient_continuous, dice_coefficient_dilated, dice_loss, BCEDiceLoss, TverskyLoss, BCETverskyLoss, numpy_dice
from src.losses import train_model

# 1. Load data

# 2. Prepare train/validation/test

# 3. Create PyTorch datasets/loaders

# 4. Create model
model = ResNet34Segmentation32(pretrained=True)

# 5. Train
model, history = train_model()

# 6. Evaluate
