import numpy as np
import cv2
import torch
from torch.utils.data import Dataset

# Crop dataset into smaller pictures (32x32) to extend dataset
def crop_dataset(dataset, crop_size, overlap = False):
    new_dataset = []
    for data in dataset:
        image = data['image']
        mask = data['mask']*1
        # Calculate the number of crops along each dimension
        num_crops_rows = image.shape[1] // crop_size[0]
        num_crops_cols = image.shape[2] // crop_size[1]

        # Iterate over rows of the image to crop
        for r in range(num_crops_rows):
            # Iterate over columns of the image to crop
            for c in range(num_crops_cols):
                # Crop the image for each channel
                output_data = {}
                output_data['coordx'] = data['coordx']
                output_data['coordy'] = data['coordy']
                output_data['mask_x'] = data['mask_x']
                output_data['mask_y'] = data['mask_y']
                output_data['crop_x'] = c
                output_data['crop_y'] = r
                output_data['dataset'] = data['dataset']
                output_data['sub_node'] = data['sub_node']
                output_data['original'] = data['original'] 
                cropped_channel_images = []
                for channel_image in image:
                    cropped_channel_image = channel_image[r * crop_size[0]:(r + 1) * crop_size[0], c * crop_size[1]:(c + 1) * crop_size[1]]
                    cropped_channel_images.append(cropped_channel_image)

                # Crop the label
                cropped_mask = mask[r * crop_size[0]:(r + 1) * crop_size[0], c * crop_size[1]:(c + 1) * crop_size[1]]

                # Append the cropped image, its label, and the original index to the dataset
                output_data['image'] = np.stack(cropped_channel_images, axis=0)
                output_data['mask'] = cropped_mask
                new_dataset.append(output_data)
                # Crop overlapping tiles
        if overlap:
            # Iterate over rows of the image to crop
            for r in range(num_crops_rows):
                # Iterate over columns of the image to crop
                for c in range(num_crops_cols-1): 
                    output_data = {}
                    output_data['coordx'] = data['coordx']
                    output_data['coordy'] = data['coordy']
                    output_data['mask_x'] = data['mask_x']
                    output_data['mask_y'] = data['mask_y']
                    output_data['crop_x'] = c
                    output_data['crop_y'] = r
                    output_data['dataset'] = data['dataset']
                    output_data['sub_node'] = data['sub_node']
                    # Brand as augmented/not original crop (allow delete during reconstruction)
                    output_data['original'] = 'False'
                    cropped_channel_images = []
                    for channel_image in image:
                        cropped_channel_image = channel_image[r * crop_size[0]:(r + 1) * crop_size[0], c * crop_size[1] + (crop_size[1]//2):(c + 1) * crop_size[1] + (crop_size[1]//2)]
                        cropped_channel_images.append(cropped_channel_image)

                    # Crop the label
                    cropped_mask = mask[r * crop_size[0]:(r + 1) * crop_size[0], c * crop_size[1] + (crop_size[1]//2):(c + 1) * crop_size[1] + (crop_size[1]//2)]

                    # Append the cropped image, its label, and the original index to the dataset
                    output_data['image'] = np.stack(cropped_channel_images, axis=0)
                    output_data['mask'] = cropped_mask
                    new_dataset.append(output_data)
            for r in range(num_crops_rows-1):
                # Iterate over columns of the image to crop
                for c in range(num_crops_cols): 
                    output_data = {}
                    output_data['coordx'] = data['coordx']
                    output_data['coordy'] = data['coordy']
                    output_data['mask_x'] = data['mask_x']
                    output_data['mask_y'] = data['mask_y']
                    output_data['crop_x'] = c
                    output_data['crop_y'] = r
                    output_data['dataset'] = data['dataset']
                    output_data['sub_node'] = data['sub_node']
                    # Brand as augmented/not original crop (allow delete during reconstruction)
                    output_data['original'] = 'False'
                    cropped_channel_images = []
                    for channel_image in image:
                        cropped_channel_image = channel_image[r * crop_size[0] + (crop_size[0]//2):(r + 1) * crop_size[0]+ (crop_size[0]//2), c * crop_size[1]:(c + 1) * crop_size[1]]
                        cropped_channel_images.append(cropped_channel_image)

                    # Crop the label
                    cropped_mask = mask[r * crop_size[0]+ (crop_size[0]//2):(r + 1) * crop_size[0]+ (crop_size[0]//2), c * crop_size[1]:(c + 1) * crop_size[1]]

                    # Append the cropped image, its label, and the original index to the dataset
                    output_data['image'] = np.stack(cropped_channel_images, axis=0)
                    output_data['mask'] = cropped_mask
                    new_dataset.append(output_data)

    return new_dataset

# Downsample mathematically the image resolution but keep same size to match mask
def downsample_dataset(dataset, res_factor, middle = 'avg'):

    if middle == None: 
        middle = 'avg'

    if (res_factor > dataset[0]['image'].shape[1]) or (res_factor < 1):
        print("Downsample factor exceeds image size or inferior to 1, returning dataset unchanged.")
        return dataset
    
    print("Downsampling resolution ...")
    new_dataset = []
    for data in dataset:
        image = data['image']
        mask = data['mask']*1
        C, H, W = image.shape

        new_H = (H // res_factor) * res_factor
        new_W = (W // res_factor) * res_factor
        image = image[:, :new_H, :new_W]
        mask = mask[:new_H, :new_W]

        output_data = {}
        output_data['coordx'] = data['coordx']
        output_data['coordy'] = data['coordy']
        output_data['mask_x'] = data['mask_x']
        output_data['mask_y'] = data['mask_y']
        output_data['dataset'] = data['dataset']
        output_data['sub_node'] = data['sub_node']
        output_data['original'] = data['original'] 
        output_data['mask'] = mask

        downsampled_channel_images = []
        for channel_image in image:
            downsampled_channel_image = np.zeros_like(channel_image)
            # Iterate over rows of the image to crop
            for r in range(0, new_H, res_factor):
                for c in range(0, new_W, res_factor):
                    block = channel_image[r:r+res_factor, c:c+res_factor]
                    if middle == 'center':
                        if res_factor % 2 == 1:
                            center = res_factor // 2
                            value = block[center, center]
                        else:
                            c1 = res_factor // 2 - 1
                            c2 = res_factor // 2
                            value = np.mean([block[c1, c1], block[c1, c2],block[c2, c1], block[c2, c2]])
                    if middle == 'corner':
                        value = block[0,0]
                    if middle == 'avg':
                        value = np.average(block)
                    if middle == 'band':
                        value = np.zeros_like(block)
                        for ind in range(res_factor):
                            if res_factor % 2 == 1:
                                center = res_factor // 2
                                value[ind,:] = block[ind, center]
                            else:
                                c1 = res_factor // 2 - 1
                                c2 = res_factor // 2
                                value[ind,:] = np.mean([block[ind, c1], block[ind, c2]])
                    downsampled_channel_image[r:r+res_factor, c:c+res_factor] = value
            downsampled_channel_images.append(downsampled_channel_image)

        output_data['image'] = np.stack(downsampled_channel_images, axis=0)
        new_dataset.append(output_data)

    print("Resolution downsampled successfully.")
    return new_dataset

# Rotate some of the image and add them to the dataset --> extend dataset
def rotate_image_and_mask(image, mask, angle):
    if angle == 90:
        rotated_image = np.array([cv2.rotate(channel, cv2.ROTATE_90_COUNTERCLOCKWISE) for channel in image])
        rotated_label = cv2.rotate(mask, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif angle == 180:
        rotated_image = np.array([cv2.rotate(channel, cv2.ROTATE_180) for channel in image])
        rotated_label = cv2.rotate(mask, cv2.ROTATE_180)
    elif angle == 270:
        rotated_image = np.array([cv2.rotate(channel, cv2.ROTATE_90_CLOCKWISE) for channel in image])
        rotated_label = cv2.rotate(mask, cv2.ROTATE_90_CLOCKWISE)
    else:
        rotated_image = image  # No rotation for angle 0
        rotated_label = mask  # No rotation for angle 0

    return rotated_image, rotated_label

def rotate_dataset(dataset, angle, cancer_threshold, aug_type = 'cancer'):
    new_dataset = []
    for data in dataset:
        # Augment margins
        if aug_type == 'margins' and np.sum(data['mask']*1) != data['mask'].size and np.sum(data['mask']*1) != 0:
            rotated_image, rotated_label = rotate_image_and_mask(data['image'], data['mask'], angle)
            # Append rotated image and its label
            output = data.copy()
            output['image'] = rotated_image
            output['mask'] = rotated_label
            output['original'] = False
            new_dataset.append(output)
        # Augment cancer dataset above a threshold
        if aug_type == 'cancer' and np.sum(data['mask'])/data['mask'].size >= cancer_threshold:
            rotated_image, rotated_label = rotate_image_and_mask(data['image'], data['mask'], angle)
            # Append rotated image and its label
            output = data.copy()
            output['image'] = rotated_image
            output['mask'] = rotated_label
            output['original'] = False
            new_dataset.append(output)
        if aug_type == 'normal' and np.sum(data['mask'])/data['mask'].size <= cancer_threshold:
            rotated_image, rotated_label = rotate_image_and_mask(data['image'], data['mask'], angle)
            # Append rotated image and its label
            output = data.copy()
            output['image'] = rotated_image
            output['mask'] = rotated_label
            output['original'] = False
            new_dataset.append(output)
    return new_dataset

# Flip some of the image and add them to the dataset --> extend dataset
def flip_image_and_mask(image, mask, direction):
    if direction == 'horizontal':
        flipped_image = np.array([cv2.flip(channel, 1) for channel in image])
        flipped_label = cv2.flip(mask, 1)
    elif direction == 'vertical':
        flipped_image = np.array([cv2.flip(channel, 0) for channel in image])
        flipped_label = cv2.flip(mask, 0)
    else:
        flipped_image = image  # No flipping
        flipped_label = mask  # No flipping

    return flipped_image, flipped_label

def flip_dataset(dataset, direction, cancer_threshold, aug_type):
    new_dataset = []
    for data in dataset:
        # Augment margins
        if aug_type == 'margin' and np.sum(data['mask']*1) != data['mask'].size and np.sum(data['mask']*1) != 0:
            flipped_image, flipped_mask = flip_image_and_mask(data['image'], data['mask'], direction)
            # Append rotated image and its label
            output = data.copy()
            output['image'] = flipped_image
            output['mask'] = flipped_mask
            output['original'] = False
            new_dataset.append(output)
        # Augment cancer dataset above a threshold
        if aug_type == 'cancer' and np.sum(data['mask'])/data['mask'].size >= cancer_threshold:
            flipped_image, flipped_mask = flip_image_and_mask(data['image'], data['mask'], direction)
            # Append rotated image and its label
            output = data.copy()
            output['image'] = flipped_image
            output['mask'] = flipped_mask
            output['original'] = False
            new_dataset.append(output)
        if aug_type == 'normal' and np.sum(data['mask'])/data['mask'].size <= cancer_threshold:
            flipped_image, flipped_mask = flip_image_and_mask(data['image'], data['mask'], direction)
            # Append rotated image and its label
            output = data.copy()
            output['image'] = flipped_image
            output['mask'] = flipped_mask
            output['original'] = False
            new_dataset.append(output)
    return new_dataset

# crop the images to get a bigger dataset and more similar FOV as with the endoscope
def get_cropped_dataset(dataset, crop, overlap):
    return crop_dataset(dataset, crop, overlap)

def get_augmented_dataset(dataset, name, aug_type, threshold = 0.5):
    print(f"Original {name} dataset {len(dataset)}", end='')
    augmented_dataset = dataset.copy()
    augmented_dataset = np.concatenate((augmented_dataset, rotate_dataset(dataset, 90, threshold, aug_type)))
    augmented_dataset = np.concatenate((augmented_dataset, flip_dataset(augmented_dataset, "vertical", threshold, aug_type)))
    print(f", Augmented {len(augmented_dataset)}.")
    return augmented_dataset

def split_dataset(dataset):
    return (np.array([data['image'] for data in dataset]), np.array([data['mask'] for data in dataset]))

def generate_64_dataset(dataset, name, threshold, crop, aug_type, overlap = False):
    if aug_type == 'none': 
        return get_cropped_dataset(dataset, crop, overlap)
    else: 
        return get_augmented_dataset(get_cropped_dataset(dataset, crop, overlap), name, aug_type, threshold)

def load_datasets(dataset, dataset_test = None, dataset_test_name = None, crop = 256, train_pct = 0.70, val_pct = 0.15, test_pct = 0.15, keep = [0, 1, 2], overlap = False, aug_type = 'cancer', thre = 0.5, res_factor = None, res_middle = None, mask_train = 0, mask_test = 0, bin_test = False, nadh = False) :

    # Merge if dataset is a list of datasets
    if (isinstance(dataset, list) and len(dataset) > 0 and isinstance(dataset[0], list)):
        print("Merging multiple datasets...")
        dataset = [item for subdataset in dataset for item in subdataset]
    
    # Merge datasets if both provided and a test name is specified
    if (dataset_test is not None) and (dataset_test_name is not None):
        print("Merging dataset and dataset_test before selecting test set...")
        dataset = list(dataset) + list(dataset_test)
        dataset_test = None  
        
    # Shuffle dataset
    np.random.shuffle(dataset)

    # Remove tiles with too many zero values in original channel 0 (lifetime NADH)
    if nadh:
        print(f"Dataset size before NADH filtering: {len(dataset)}")
        dataset = [d for d in dataset if np.mean(d['image'][0] == 0) <= 0.25]
        if dataset_test is not None:
            print(f"Dataset test size before NADH filtering: {len(dataset)}")
            dataset_test = [d for d in dataset_test if np.mean(d['image'][0] == 0) <= 0.25]
        print(f"Dataset size after NADH filtering: {len(dataset)}")
        if dataset_test is not None:
            print(f"Test dataset size after NADH filtering: {len(dataset_test)}")

    # Augmentation pattern: augment cancer part or margin images part
    aug_type = aug_type #'cancer', 'none', 'normal'
    thre = thre

    # Select test set manually
    if (dataset_test is None) and (dataset_test_name is not None):
        # If a single string is provided, convert to list
        if isinstance(dataset_test_name, str):
            dataset_test_name = [dataset_test_name]
        # Gather all requested datasets into test set
        dataset_test = [d for d in dataset if d['dataset'] in dataset_test_name]
        # Keep remaining datasets for training
        dataset = [d for d in dataset if d['dataset'] not in dataset_test_name]
        print(f"Using {dataset_test_name} as test set")

    # Compute split indices
    n_total = len(dataset)
    # dataset_test available
    n_train = int(n_total * 0.8) #n_train = int(n_total * train_pct)
    # mixed testing
    n_val = int(n_total*train_pct) #n_val = int(n_total * val_pct)
    n_test = int(n_total*train_pct + n_total*val_pct)

    # Print datasets
    datasets = list({d['dataset'] for d in dataset})
    print("The datasets included in the training are: ", datasets)

    for a in range(len(dataset)):
        # Compute ratio intensity channel
        img = dataset[a]['image']
        new_channels = []
        new_channels.append(img[1])
        new_channels.append(img[3])
        if bin_test == False: 
            ratio = np.zeros_like(img[0])
            np.divide(img[0],img[2],out=ratio,where=img[2] != 0)
        else: 
            ratio = np.zeros_like(img[0])
            np.divide((img[0]-img[2]),(img[0]+img[2]),out=ratio,where=(img[0]+img[2]) != 0)
        new_channels.append(ratio)
        dataset[a]['image'] = np.stack(new_channels)
        # Clip outliers
        vmin1, vmax1 = np.percentile(dataset[a]['image'][0], (1, 99))
        vmin2, vmax2 = np.percentile(dataset[a]['image'][1], (1, 99))
        vmin3, vmax3 = np.percentile(dataset[a]['image'][2], (1, 99))
        dataset[a]['image'][0] = np.clip(dataset[a]['image'][0], vmin1, vmax1)
        dataset[a]['image'][1] = np.clip(dataset[a]['image'][1], vmin2, vmax2)
        dataset[a]['image'][2] = np.clip(dataset[a]['image'][2], vmin3, vmax3)
        img = dataset[a]['image']
        new_channels = []
        for item in keep:
            # Normal behavior: keep=[0,1]
            if isinstance(item, int):
                new_channels.append(img[item])
            # Multiplication behavior: keep=['0*1','0*2']
            elif isinstance(item, str):
                channel_ids = [int(x) for x in item.split('*')]
                out = img[channel_ids[0]].copy()
                for c in channel_ids[1:]:
                    out *= img[c]
                new_channels.append(out)
        dataset[a]['image'] = np.stack(new_channels)
        dataset[a]['mask'] = np.array(dataset[a]['mask'][mask_train])

    if dataset_test is not None: 
        for a in range(len(dataset_test)):
            # Compute ratio intensity channel
            img = dataset_test[a]['image']
            new_channels = []
            new_channels.append(img[1])
            new_channels.append(img[3])
            if bin_test == False: 
                ratio = np.zeros_like(img[0])
                np.divide(img[0],img[2],out=ratio,where=img[2] != 0)
            else: 
                ratio = np.zeros_like(img[0])
                np.divide((img[0]-img[2]),(img[0]+img[2]),out=ratio,where=(img[0]+img[2]) != 0)
            new_channels.append(ratio)
            dataset_test[a]['image'] = np.stack(new_channels)
            # Clip outliers
            vmin1, vmax1 = np.percentile(dataset_test[a]['image'][0], (1, 99))
            vmin2, vmax2 = np.percentile(dataset_test[a]['image'][1], (1, 99))
            vmin3, vmax3 = np.percentile(dataset_test[a]['image'][2], (1, 99))
            dataset_test[a]['image'][0] = np.clip(dataset_test[a]['image'][0], vmin1, vmax1)
            dataset_test[a]['image'][1] = np.clip(dataset_test[a]['image'][1], vmin2, vmax2)
            dataset_test[a]['image'][2] = np.clip(dataset_test[a]['image'][2], vmin3, vmax3)
            img = dataset_test[a]['image']
            new_channels = []
            for item in keep:
                # Normal behavior: keep=[0,1]
                if isinstance(item, int):
                    new_channels.append(img[item])
                # Multiplication behavior: keep=['0*1','0*2']
                elif isinstance(item, str):
                    channel_ids = [int(x) for x in item.split('*')]
                    out = img[channel_ids[0]].copy()
                    for c in channel_ids[1:]:
                        out *= img[c]
                    new_channels.append(out)
            dataset_test[a]['image'] = np.stack(new_channels)
            dataset_test[a]['mask'] = np.array(dataset_test[a]['mask'][mask_test])

    # Decrease resolution
    if res_factor is not None: 
        dataset = downsample_dataset(dataset, res_factor, res_middle)
        if dataset_test is not None: 
            dataset_test = downsample_dataset(dataset_test, res_factor, res_middle)

    # Split and augment (crop and rotate) the dataset
    if dataset_test is not None: 
        train_ds = generate_64_dataset(dataset[:n_train], "train", thre, crop, aug_type, overlap = overlap)
        validation_ds = generate_64_dataset(dataset[n_train:], 'valid', thre, crop, aug_type, overlap = overlap)
        test_ds = generate_64_dataset(dataset_test, 'test', 0.0, crop, 'none')
    else: 
        print('Mixed testing mode.')
        train_ds = generate_64_dataset(dataset[:n_val], "train", thre, crop, aug_type, overlap = overlap)
        validation_ds = generate_64_dataset(dataset[n_val:n_test], 'valid', thre, crop, aug_type, overlap = overlap)
        test_ds = generate_64_dataset(dataset[n_test:], 'test', 0.0, crop, 'none')

    print(f"Total (before crop): {n_total}, (After crop) Train: {len(train_ds)}, Validation: {len(validation_ds)}, Test: {len(test_ds)}")

    # Count amount of cancer/healthy images in train, validation and test datasets
    maj_true, maj_false, maj_any, all_zero, all_one = 0, 0, 0, 0, 0
    for i in train_ds:
        mask = i['mask']
        if mask.any():
            maj_any += 1
        if np.sum(mask) == 0:
            all_zero += 1
        if np.sum(mask) == mask.size:
            all_one += 1
        if np.sum(mask) >= (mask.size / 2):
            maj_true += 1
        if np.sum(mask) < (mask.size / 2):
            maj_false += 1
    print(f'For train dataset: \nContaining Cancer images: {maj_any}\nMajority of Cancer images: {maj_true}\nMajority Healthy images: {maj_false}\nAll Cancer images: {all_one}\nAll Healthy images: {all_zero}')
    maj_true, maj_false, maj_any, all_zero, all_one = 0, 0, 0, 0, 0
    for i in validation_ds:
        mask = i['mask']
        if mask.any():
            maj_any += 1
        if np.sum(mask) == 0:
            all_zero += 1
        if np.sum(mask) == mask.size:
            all_one += 1
        if np.sum(mask) >= (mask.size / 2):
            maj_true += 1
        if np.sum(mask) < (mask.size / 2):
            maj_false += 1
    print(f'For validation dataset: \nContaining Cancer images: {maj_any}\nMajority of Cancer images: {maj_true}\nMajority Healthy images: {maj_false}\nAll Cancer images: {all_one}\nAll Healthy images: {all_zero}')
    maj_true, maj_false, maj_any, all_zero, all_one = 0, 0, 0, 0, 0
    for i in test_ds:
        mask = i['mask']
        if mask.any():
            maj_any += 1
        if np.sum(mask) == 0:
            all_zero += 1
        if np.sum(mask) == mask.size:
            all_one += 1
        if np.sum(mask) >= (mask.size / 2):
            maj_true += 1
        if np.sum(mask) < (mask.size / 2):
            maj_false += 1
    print(f'For test dataset: \nContaining Cancer images: {maj_any}\nMajority of Cancer images: {maj_true}\nMajority Healthy images: {maj_false}\nAll Cancer images: {all_one}\nAll Healthy images: {all_zero}')
    
    return train_ds, validation_ds, test_ds

class CancerDataset(Dataset):
    def __init__(self, dataset, include_info=False):
        self.dataset = dataset
        self.include_info = include_info

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        # Convert image and mask to torch tensors
        # image shape: (channels, height, width)
        image = torch.tensor(data['image'], dtype=torch.float32) # (n, 3, 32, 32)
        mask = torch.tensor(data['mask'], dtype=torch.float32).unsqueeze(0)  # add channel dim (n, 1, 32, 32)
        if self.include_info:
            info_dict = {
                "dataset": data['dataset'],
                "sub_node": data['sub_node'],
                "coordx": data['coordx'],
                "coordy": data['coordy'], 
                "mask_x": data['mask_x'],
                "mask_y": data['mask_y'], 
                "original": data['original'],
                "crop_x": data['crop_x'],
                "crop_y": data['crop_y']
                }
            return image, mask, info_dict
        else: 
            return image, mask

def custom_collate(batch):
    inputs, labels, infos = zip(*batch)
    inputs = torch.stack(inputs)
    labels = torch.stack(labels)
    return inputs, labels, list(infos)  # keep infos as list of dicts
