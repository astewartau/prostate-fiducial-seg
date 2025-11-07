#%%
print("In Python")

# replace print so it always flushes
def print(*args, **kwargs):
    __builtins__.print(*args, **kwargs, flush=True)

print("Importing libraries...")

import glob
import os
import time
import datetime
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

import pandas as pd
import numpy as np
import torchio as tio
import scipy.ndimage
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold
#%%

# Check command-line arguments; set default if not provided.
if not len(sys.argv) == 3:
    sys.argv = ["", "T1", "0"]

print(f"=== {sys.argv[1]}-{sys.argv[2]} ===")
model_name = sys.argv[1]
fold_id = int(sys.argv[2])
random_state = 42
timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y%m%d-%H%M%S')

batch_size = 6
training_epochs = 1300
lr = 0.003
num_classes = 3
ce_loss_weights = torch.tensor([1, 5, 1], dtype=torch.float32)

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print(f"GPU is available. Using GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("GPU is NOT available. Using CPU.")

# --- Data preparation: Build dataframe from bids directories ---
bids_dirs = ["bids-2025-2"]

# Setup dictionary to hold paths for each subject
# This will be used to create a pandas dataframe later
# Initialize an empty list to hold the paths
paths_list = []

for bids_dir in bids_dirs:
    subject_dirs = sorted(glob.glob(os.path.join(bids_dir, "z*")))

    for subject_dir in subject_dirs:
        subject_id = os.path.basename(subject_dir)

        # Check if roi_niftis_mri_space/ directory exists
        roi_dir = os.path.join(subject_dir, "roi_niftis_mri_space")
        if not os.path.exists(roi_dir):
            print(f"Skipping {subject_id} as roi_niftis_mri_space directory does not exist.")
            continue

        # Get all nifti files
        nii_paths = glob.glob(os.path.join(subject_dir, "*.nii")) + \
                    glob.glob(os.path.join(subject_dir, "*.nii.gz"))

        # Get all nifti files in the roi_niftis_mri_space directory
        nii_paths += glob.glob(os.path.join(roi_dir, "*.nii")) + \
                     glob.glob(os.path.join(roi_dir, "*.nii.gz"))

        paths_dict = {}

        for nii_path in nii_paths:
            paths_dict['subject_id'] = subject_id
            file_name = os.path.basename(nii_path).split('.')[0]

            if 'GS1' in file_name:
                segmentation_name = "GS1"
            elif 'GS2' in file_name:
                segmentation_name = "GS2"
            elif 'GS3' in file_name:
                segmentation_name = "GS3"
            else:
                segmentation_name = "_".join(file_name.split('_')[1:])

            if segmentation_name == "roi_CTV_High_MR":
                segmentation_name = "Prostate"
            
            paths_dict[segmentation_name] = nii_path

        # Append the dictionary to the list
        paths_list.append(paths_dict)

# Convert the list of dictionaries to a pandas DataFrame
df = pd.DataFrame(paths_list)
# Replace NaN values with None
df = df.where(pd.notnull(df), None)

# For rows with no Prostate segmentation, if there is a roi_CTV_Low_MR segmentation, use it as Prostate
for index, row in df.iterrows():
    if row.get('Prostate') is None and row.get('roi_CTV_Low_MR') is not None:
        df.at[index, 'Prostate'] = row['roi_CTV_Low_MR']

for index, row in df.iterrows():
    if row.get('Prostate') is None and row.get('roi_CTVp_MR') is not None:
        df.at[index, 'Prostate'] = row['roi_CTVp_MR']

# Remove all columns except
columns_to_keep = ['subject_id', 'MRI', 'MRI_homogeneity-corrected', 'CT', 'seeds', 'Prostate']
df = df[columns_to_keep]

# Keep only rows that have both the input and segmentation files.
infile_cols = ['MRI_homogeneity-corrected']
seg_col = 'seeds'

# Remove rows with missing values in the required columns
for col in infile_cols + [seg_col]:
    if col not in df.columns:
        raise ValueError(f"Column {col} not found in DataFrame.")
    df = df[df[col].notna()].reset_index(drop=True)

print(f"Number of subjects: {len(df)}")
print(f"infile_cols: {infile_cols}; segmentation column: {seg_col}")

#%%
# --- KFold splitting ---
k_folds = len(df)
kf = KFold(n_splits=k_folds, random_state=random_state, shuffle=True)
(train_index, valid_index) = list(kf.split(df))[fold_id]

# Assert that there is exactly one validation example
assert len(valid_index) == 1, f"Expected one validation subject, but got {len(valid_index)}."
print(f"Number of training examples: {len(train_index)}")
print(f"Number of validation examples: {len(valid_index)}")

# --- Custom transform: Random Intensity Shift using Gamma Correction for selected modalities only ---
class RandomIntensityShift(tio.Transform):
    def __init__(self, gamma_range=(0.8, 1.2), p=0.5, apply_to_keys=None):
        super().__init__()
        self.gamma_range = gamma_range
        self.p = p
        self.apply_to_keys = apply_to_keys  # List of keys to apply the transform

    def apply_transform(self, subject):
        if torch.rand(1).item() > self.p:
            return subject

        # If keys to modify are provided, limit the transformation to those keys.
        keys_to_transform = self.apply_to_keys if self.apply_to_keys is not None else list(subject.keys())
        for image_name in keys_to_transform:
            if image_name in subject and isinstance(subject[image_name], tio.ScalarImage):
                img_tensor = subject[image_name].data
                if img_tensor.ndim == 3:
                    img_tensor = img_tensor.unsqueeze(0)
                gamma = np.random.uniform(*self.gamma_range)
                img_tensor = img_tensor.pow(gamma)
                subject[image_name].set_data(img_tensor)
        return subject

# --- Custom transform: Merge Input Channels ---
class MergeInputChannels(tio.Transform):
    def __init__(self, infile_cols):
        super().__init__()
        self.infile_cols = infile_cols
    def apply_transform(self, subject):
        input_tensors = []
        for col in self.infile_cols:
            if col not in subject:
                continue
            tensor = subject[col].data
            if tensor.ndim == 3:
                tensor = tensor.unsqueeze(0)
            input_tensors.append(tensor)
        if not input_tensors:
            return subject
        merged = torch.cat(input_tensors, dim=0)
        subject['image'] = tio.ScalarImage(tensor=merged, affine=subject[self.infile_cols[0]].affine)
        subject['mask'] = subject['seg']
        return subject

# --- Build training transforms ---
class PadToCompatibleSize(tio.Transform):
    def __init__(self, min_factor=32):
        super().__init__()
        self.min_factor = min_factor

    def apply_transform(self, subject):
        for key in subject.keys():
            if isinstance(subject[key], (tio.ScalarImage, tio.LabelMap)):
                vol = subject[key].data
                target_shape = []
                for dim in vol.shape:
                    compatible_dim = ((dim + self.min_factor - 1) // self.min_factor) * self.min_factor
                    target_shape.append(compatible_dim)
                target_shape = tuple(target_shape)
                if vol.shape != target_shape:
                    padded_vol = F.pad(vol, [0, target_shape[2] - vol.shape[2],
                                             0, target_shape[1] - vol.shape[1],
                                             0, target_shape[0] - vol.shape[0]])
                    # Preserve the original type (ScalarImage vs LabelMap)
                    if isinstance(subject[key], tio.ScalarImage):
                        subject[key] = tio.ScalarImage(tensor=padded_vol, affine=subject[key].affine)
                    else:  # LabelMap
                        subject[key] = tio.LabelMap(tensor=padded_vol, affine=subject[key].affine)
        return subject

if "T1" in model_name:
    training_transforms = tio.Compose([
        RandomIntensityShift(gamma_range=(0.8, 1.2), p=0.75, apply_to_keys=['MRI_homogeneity-corrected']),
        PadToCompatibleSize(min_factor=32),
        tio.RandomFlip(axes=(0, 1)),
        tio.RandomAffine(degrees=(90, 90, 90)),
        tio.ZNormalization(),
        MergeInputChannels(infile_cols)
    ])
else:
    training_transforms = tio.Compose([
        PadToCompatibleSize(min_factor=32),
        tio.RandomFlip(axes=(0, 1)),
        tio.RandomAffine(degrees=(90, 90, 90)),
        tio.ZNormalization(),
        MergeInputChannels(infile_cols)
    ])

# --- Build TorchIO subjects ---
subjects = []
for _, row in df.iterrows():
    subject_dict = {}
    for col in infile_cols:
        subject_dict[col] = tio.ScalarImage(row[col])
    subject_dict['seg'] = tio.LabelMap(row[seg_col])
    subject = tio.Subject(**subject_dict)
    subjects.append(subject)

subjects_train = [subjects[i] for i in train_index]
subjects_valid = [subjects[i] for i in valid_index]

# Extract validation example metadata
validation_row = df.iloc[valid_index[0]]
validation_metadata = {
    'subject_id': validation_row['subject_id'],
    'fold_id': fold_id,
    'validation_index': int(valid_index[0]),
    'input_files': {},
    'segmentation_file': os.path.abspath(validation_row[seg_col])
}
for col in infile_cols:
    validation_metadata['input_files'][col] = os.path.abspath(validation_row[col])

print(f"\nValidation subject: {validation_metadata['subject_id']}")
print(f"Validation input file(s):")
for col, path in validation_metadata['input_files'].items():
    print(f"  {col}: {path}")
print(f"Validation segmentation file: {validation_metadata['segmentation_file']}\n")

train_subjects_dataset = tio.SubjectsDataset(subjects_train, transform=training_transforms)
valid_subjects_dataset = tio.SubjectsDataset(subjects_valid, transform=training_transforms)

class TorchIODatasetWrapper(torch.utils.data.Dataset):
    def __init__(self, subjects_dataset):
        self.subjects_dataset = subjects_dataset
    def __len__(self):
        return len(self.subjects_dataset)
    def __getitem__(self, index):
        subject = self.subjects_dataset[index]
        if 'image' not in subject:
            raise KeyError("Missing 'image' key in subject")
        image = subject['image'].data
        if 'mask' not in subject:
            raise KeyError("Missing 'mask' key in subject")
        mask = subject['mask'].data.squeeze(0)
        mask = mask.long()
        return image, mask

train_dataset = TorchIODatasetWrapper(train_subjects_dataset)
valid_dataset = TorchIODatasetWrapper(valid_subjects_dataset)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

# --- 3D UNet model definition ---
class UNet3D(nn.Module):
    def __init__(self, in_channels, out_channels, channels=(16, 32, 64, 128, 256)):
        super(UNet3D, self).__init__()
        self.encoders = nn.ModuleList()
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        prev_channels = in_channels
        for ch in channels:
            self.encoders.append(self.conv_block(prev_channels, ch))
            prev_channels = ch
        self.bottleneck = self.conv_block(prev_channels, prev_channels * 2)
        rev_channels = list(reversed(channels))
        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        cur_channels = prev_channels * 2
        for ch in rev_channels:
            self.upconvs.append(nn.ConvTranspose3d(cur_channels, ch, kernel_size=2, stride=2))
            self.decoders.append(self.conv_block(ch * 2, ch))
            cur_channels = ch
        self.final_conv = nn.Conv3d(cur_channels, out_channels, kernel_size=1)
    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        enc_feats = []
        for encoder in self.encoders:
            x = encoder(x)
            enc_feats.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        for upconv, decoder, enc in zip(self.upconvs, self.decoders, reversed(enc_feats)):
            x = upconv(x)
            if x.shape != enc.shape:
                diffZ = enc.size()[2] - x.size()[2]
                diffY = enc.size()[3] - x.size()[3]
                diffX = enc.size()[4] - x.size()[4]
                x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                              diffY // 2, diffY - diffY // 2,
                              diffZ // 2, diffZ - diffZ // 2])
            x = torch.cat([enc, x], dim=1)
            x = decoder(x)
        return self.final_conv(x)

model = UNet3D(in_channels=len(infile_cols), out_channels=num_classes).to(device)

ce_loss_fn = nn.CrossEntropyLoss(weight=ce_loss_weights.to(device))
def dice_loss(pred, target, eps=1e-5):
    pred = F.softmax(pred, dim=1)
    target_onehot = F.one_hot(target, num_classes=pred.shape[1]).permute(0, 4, 1, 2, 3).float()
    intersection = (pred * target_onehot).sum(dim=(2,3,4))
    sums = pred.sum(dim=(2,3,4)) + target_onehot.sum(dim=(2,3,4))
    dice = (2 * intersection + eps) / (sums + eps)
    return 1 - dice.mean()
def combined_loss(pred, target):
    return ce_loss_fn(pred, target) + 0.5 * dice_loss(pred, target)

def process_volume(pred_vol, targ_vol, structure):
    _, pred_nlabels = scipy.ndimage.label(pred_vol, structure=structure)
    _, targ_nlabels = scipy.ndimage.label(targ_vol, structure=structure)

    overlap = np.logical_and(pred_vol == targ_vol, pred_vol == 1)
    _, n_overlaps = scipy.ndimage.label(overlap, structure=structure)

    return pred_nlabels, targ_nlabels, n_overlaps

def compute_metrics(pred, targ):
    structure = np.ones((3, 3, 3), dtype=bool)
    total_pred_marker_count = 0
    total_targ_marker_count = 0
    total_overlap_count = 0

    pred_marker = (pred == 1).astype(np.int32)
    targ_marker = (targ == 1).astype(np.int32)

    # Optional dilation to account for small misalignments
    pred_marker = scipy.ndimage.binary_dilation(pred_marker)
    targ_marker = scipy.ndimage.binary_dilation(targ_marker)

    for i in range(pred_marker.shape[0]):
        p_vol = pred_marker[i]
        t_vol = targ_marker[i]
        p_n, t_n, n_overlap = process_volume(p_vol, t_vol, structure)
        total_pred_marker_count += p_n
        total_targ_marker_count += t_n
        total_overlap_count += n_overlap

    false_negative = total_targ_marker_count - total_overlap_count
    false_positive = total_pred_marker_count - total_overlap_count

    return {
         "actual_markers": total_targ_marker_count,
         "true_positive": total_overlap_count,
         "false_negative": false_negative,
         "false_positive": false_positive
    }


train_loss_list = []
valid_loss_list = []

optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=training_epochs)

best_valid_loss = float('inf')
epochs_no_improve = 0
patience = 500

#%%
start_time = time.time()
for epoch in range(training_epochs):
    model.train()
    train_loss = 0
    for images, masks in train_loader:
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = combined_loss(outputs, masks)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * images.size(0)
    train_loss /= len(train_loader.dataset)
    
    model.eval()
    valid_loss = 0
    all_preds = []
    all_targs = []
    with torch.no_grad():
        for images, masks in valid_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss = combined_loss(outputs, masks)
            valid_loss += loss.item() * images.size(0)
            preds = torch.argmax(outputs, dim=1)
            all_preds.append(preds.cpu().numpy())
            all_targs.append(masks.cpu().numpy())
    valid_loss /= len(valid_loader.dataset)
    all_preds = np.concatenate(all_preds, axis=0)
    all_targs = np.concatenate(all_targs, axis=0)
    
    metrics = compute_metrics(all_preds, all_targs)
    
    train_loss_list.append(train_loss)
    valid_loss_list.append(valid_loss)
    
    print(f"Epoch {epoch+1}/{training_epochs}, Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}")
    print(f"Actual seed markers: {metrics['actual_markers']}, True positive: {metrics['true_positive']}, "
        f"False negative: {metrics['false_negative']}, False positive: {metrics['false_positive']}")

    plt.figure()
    plt.plot(range(1, epoch+2), train_loss_list, label="Train Loss")
    plt.plot(range(1, epoch+2), valid_loss_list, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.ylim(0, 1)    
    plt.title(f"Loss Curve for {model_name}")
    plt.legend()
    plt.savefig(f"{model_name}-{timestamp}-{fold_id}-loss.png")
    plt.close()
    
    if valid_loss < best_valid_loss - 0.01:
        best_valid_loss = valid_loss
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'validation_metadata': validation_metadata,
            'epoch': epoch + 1,
            'best_valid_loss': valid_loss,
            'model_name': model_name,
            'timestamp': timestamp
        }
        torch.save(checkpoint, f"{model_name}-{timestamp}-{fold_id}-best.pth")
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
    if epochs_no_improve >= patience:
        print("Early stopping triggered")
        break
    scheduler.step()

end_time = time.time()
duration_mins = (end_time - start_time) / 60
print(f"Finished training after {round(duration_mins, 2)} mins")
final_checkpoint = {
    'model_state_dict': model.state_dict(),
    'validation_metadata': validation_metadata,
    'final_epoch': epoch + 1,
    'training_duration_mins': duration_mins,
    'model_name': model_name,
    'timestamp': timestamp
}
torch.save(final_checkpoint, f"{model_name}-{timestamp}-{fold_id}-final.pth")

# %%
