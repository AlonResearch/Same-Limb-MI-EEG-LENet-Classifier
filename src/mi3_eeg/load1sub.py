from pathlib import Path
import numpy as np
import scipy.io as scio
from mi3_eeg.dataset import load_mat_from_derivatives, create_data_loader

print("=" * 80)
print("Testing Converted File")
print("=" * 80)

# Test loading the converted file
converted_file = Path("G:/My Drive/ML/dataset/sub-017_eeg200hz.mat")

print(f"\nLoading converted file: {converted_file}")

# Load using our dataset loader (should detect standardized format)
data_bundle = load_mat_from_derivatives(
    mat_path=converted_file,
    reduce_rest_ratio=1.0,
    expected_sampling_rate=200,
    validate_timepoints=True,
)

print("\n" + "=" * 80)
print("Data Bundle Information")
print("=" * 80)
print(f"Data shape: {data_bundle.data.shape}")
print(f"Labels shape: {data_bundle.labels.shape}")
print(f"Channels: {data_bundle.channel_count}")
print(f"Classes: {data_bundle.num_classes}")
print(f"Sample rate: {data_bundle.sample_rate}Hz")
print(f"Class distribution: {data_bundle.class_distribution}")

# Verify labels are correct (0, 1, 2)
unique_labels = np.unique(data_bundle.labels)
print(f"\nUnique label values: {unique_labels}")
print(f"Expected: [0 1 2] (Rest=0, Elbow=1, Hand=2)")
assert np.array_equal(unique_labels, [0, 1, 2]), "Labels don't match expected!"

# Test creating dataloader
print("\n" + "=" * 80)
print("Testing DataLoader Creation")
print("=" * 80)

dataloader = create_data_loader(
    data=data_bundle.data,
    labels=data_bundle.labels,
    batch_size=32,
    shuffle=True,
    device="cpu"
)

print(f"DataLoader created with {len(dataloader)} batches")

# Test iterating through a few batches
for i, (batch_data, batch_labels) in enumerate(dataloader):
    print(f"Batch {i}: data shape={batch_data.shape}, labels shape={batch_labels.shape}")
    print(f"  Labels in batch: {np.unique(batch_labels.cpu().numpy())}")
    if i >= 2:
        break

print("\n" + "=" * 80)
print("ALL TESTS PASSED!")
print("=" * 80)
print(f"\nSummary:")
print(f"  - Converted file loaded successfully")
print(f"  - Labels are correct: 0 (Rest), 1 (Elbow), 2 (Hand)")
print(f"  - Sampling rate: {data_bundle.sample_rate}Hz")
print(f"  - Total trials: {data_bundle.data.shape[0]}")
print(f"  - Balanced distribution: {data_bundle.class_distribution}")