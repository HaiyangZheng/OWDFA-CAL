import os
from copy import deepcopy

import numpy as np
import cv2
import dlib

from PIL import Image
from loguru import logger

from torch.utils.data import Dataset

from .face_cropper import dlib_crop_face
from .data_utils import dataset_stats


# -------------------------------------------------
# # OWDFADataset
# -------------------------------------------------
# Note that real-face images from different sources (e.g., Celeb-DF and FaceForensics++) 
# are intentionally merged and treated as one class.
def prepare_owdfa_samples(
    root,
    train=True,
    test_ratio=0.2,
    known_classes=None,
    train_classes=None,
    seed=2025
):
    """
    Prepare OWDFA samples by class-wise sampling and train/test splitting.

    The function performs the following steps:
      1. Data preparation: traverse subfolders under `root`, parse class labels
         from folder names, recursively collect image paths, and organize them
         by label (including special handling for the real-face class).
      2. Class-wise sampling: apply different sampling budgets for known and
         unknown classes, with a special rule for label 0.
      3. Train/test split: split sampled data into training and testing subsets
         according to `test_ratio`, and return the requested split based on the
         `train` flag.

    Args:
        root (str): Root directory containing class-wise subfolders.
        train (bool): If True, return training samples; otherwise return testing samples.
        test_ratio (float): Proportion of samples used for testing.
        known_classes (list[int] or None): List of known class labels.
        train_classes (list[int] or None): If provided, only these classes are included.
        seed (int): Random seed for reproducible sampling.

    Returns:
        list[list]: A list of [image_path, label] pairs.
    """

    # ---------------------------------------------------------------------
    # Initialization
    # ---------------------------------------------------------------------
    np.random.seed(seed)
    samples_by_label = {}   # label -> list of image paths

    # ---------------------------------------------------------------------
    # Step 0: Traverse folders and collect image paths per label
    # ---------------------------------------------------------------------
    for folder in os.listdir(root):
        folder_path = os.path.join(root, folder)
        if not os.path.isdir(folder_path):
            continue

        # Parse class label from folder name prefix
        try:
            label = int(folder.split('_')[0])
        except Exception as e:
            logger.info(f"Skip folder '{folder}': cannot parse label ({e})")
            continue

        # Recursively collect all images under the folder
        imgs = []
        for dirpath, _, filenames in os.walk(folder_path):
            imgs.extend([
                os.path.join(dirpath, f)
                for f in filenames
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))
            ])

        print(f"[Collect] Label {label} | Folder '{folder}' | Images found: {len(imgs)}")

        # Special handling for real-face class (label == 0):
        if label == 0:
            imgs = np.random.choice(imgs, size=10000, replace=False).tolist()

        # Merge images into label-level container
        if label not in samples_by_label:
            samples_by_label[label] = []
        samples_by_label[label].extend(imgs)

    # ---------------------------------------------------------------------
    # Summary: number of collected images per label
    # ---------------------------------------------------------------------
    label_stats = {label: len(imgs) for label, imgs in samples_by_label.items()}
    print("[Summary] Total images per label:", label_stats)

    # ---------------------------------------------------------------------
    # Step 1 & 2: Class-wise sampling and train/test split
    # ---------------------------------------------------------------------
    final_samples = []

    for label, paths in samples_by_label.items():

        # Optionally restrict to selected training classes
        if train_classes is not None and label not in train_classes:
            continue

        paths = np.array(paths)

        # -------------------------------------------------------------
        # Step 1: Determine sampling budget per class
        # -------------------------------------------------------------
        if known_classes is not None and label in known_classes:
            if label == 0:
                sampled_n = 20000   # special case: real-face class
            else:
                sampled_n = 2000
        else:
            sampled_n = 1500

        # Apply sampling if necessary
        if len(paths) > sampled_n:
            paths = np.random.choice(paths, size=sampled_n, replace=False)

        # -------------------------------------------------------------
        # Step 2: Train / Test split
        # -------------------------------------------------------------
        np.random.shuffle(paths)
        n = len(paths)
        split_idx = int(n * (1 - test_ratio))

        train_paths = paths[:split_idx]
        test_paths = paths[split_idx:]

        selected_paths = train_paths if train else test_paths
        for p in selected_paths:
            final_samples.append([p, label])

    return final_samples

class OWDFADataset(Dataset):
    def __init__(
        self,
        root,
        transform=None,
        target_transform=None,
        train=True,
        test_ratio=0.2,
        known_classes=None,
        train_classes=None,
        crop_face=True,
        predictor_path=None,
        seed=2025,
    ):
        """
        OWDFADataset: a dataset wrapper that prepares OWDFA samples and loads images.

        Args:
            root (str):
                Root directory of the dataset. Subfolders are expected to be class-wise.
            transform:
                Optional transform (e.g., Albumentations). Expected to be callable on `img`.
            target_transform:
                Optional transform applied to the label.
            train (bool):
                If True, use the training split; otherwise use the testing split.
            test_ratio (float):
                Ratio of samples used for testing.
            known_classes (list[int] or None):
                List of known class labels (used by sampling routine).
            train_classes (list[int] or None):
                If provided, only these classes are included.
            crop_face (bool):
                Whether to crop faces using dlib before applying transforms.
            predictor_path (str or None):
                Path to dlib 68-landmark predictor. Required when crop_face=True.
            seed (int):
                Random seed used by the sampling routine.
        """

        # Prepare samples (path, label) using the dataset split logic
        self.samples = prepare_owdfa_samples(
            root=root,
            train=train,
            test_ratio=test_ratio,
            known_classes=known_classes,
            train_classes=train_classes,
            seed=seed,
        )

        # Ensure targets are integers
        self.samples = [[s[0], int(s[1])] for s in self.samples]

        # Log one sample for sanity check (useful for debugging / release)
        logger.info(f"[Dataset] Example sample: {self.samples[0]}")

        # Cache targets for downstream utilities
        self.targets = [s[1] for s in self.samples]

        # Store transforms
        self.transform = transform
        self.target_transform = target_transform

        # Face-cropping configuration
        self.crop_face = crop_face
        self.predictor_path = predictor_path

        # Unique indices (often used by SSL / clustering pipelines)
        self.uq_idxs = np.arange(len(self.samples))

        # Initialize dlib components if face cropping is enabled
        if self.crop_face:
            assert predictor_path is not None, "predictor_path must be provided when crop_face=True"
            self.detector = dlib.get_frontal_face_detector()
            self.predictor = dlib.shape_predictor(predictor_path)

    def __len__(self):
        """Return the number of samples in the current split."""
        return len(self.samples)

    def load_img(self, path):
        """
        Load an image from disk, optionally crop the face, and convert BGR (OpenCV) to RGB (PIL).

        Args:
            path (str): Absolute path to an image.

        Returns:
            PIL.Image: Loaded (and optionally face-cropped) image in RGB format.
        """
        img = cv2.imread(path)
        if self.crop_face:
            img = dlib_crop_face(img, self.detector, self.predictor, align=False, margin=1.2)
        img = img[:, :, ::-1]  # BGR -> RGB
        return Image.fromarray(img)

    def __getitem__(self, idx):
        """
        Fetch one item.

        Returns a dict with:
            - image: transformed image object
            - target: label (possibly transformed)
            - idx: unique index
            - img_path: original image path
        """
        path, label = self.samples[idx]
        try:
            img = self.load_img(path)

            if self.transform is not None:
                transformed = self.transform(img)
                
                img = transformed if isinstance(transformed, list) else transformed

            if self.target_transform is not None:
                label = self.target_transform(label)

            item = {
                'image': img,
                'target': label,
                'idx': self.uq_idxs[idx],
                'img_path': path
            }
        except Exception as e:
            raise RuntimeError(f"⚠️ Failed to load image: {e} | Path: {path}")

        return item


# -------------------------------------------------
# Open-World Dataset Partitioning and Subsampling
# -------------------------------------------------

# Subsample a dataset by selecting a subset of indices.
def subsample_dataset(dataset, idxs):
    # Create a boolean mask indicating which indices are kept
    mask = np.zeros(len(dataset), dtype=bool)
    mask[idxs] = True

    # Apply the mask to samples, targets, and unique indices
    dataset.samples = np.array(dataset.samples)[mask].tolist()
    dataset.targets = np.array(dataset.targets)[mask].tolist()
    dataset.uq_idxs = dataset.uq_idxs[mask]

    # Ensure labels stored in samples are of type int
    dataset.samples = [[x[0], int(x[1])] for x in dataset.samples]

    return dataset

# Subsample a dataset to include only specified classes and remap labels.
def subsample_classes(dataset, include_classes):
    cls_idxs = [i for i, l in enumerate(dataset.targets) if l in include_classes]

    # Define a label remapping: original_label -> new_contiguous_label
    target_xform_dict = {k: i for i, k in enumerate(include_classes)}

    # Subsample the dataset to keep only selected classes
    dataset = subsample_dataset(dataset, cls_idxs)

    # Apply the target transform to map labels into a contiguous space
    dataset.target_transform = lambda x: target_xform_dict[x]

    return dataset

# Randomly select a subset of instance indices from a dataset.
def subsample_instances(dataset, prop_indices_to_subsample=0.8):
    num = int(len(dataset.targets) * prop_indices_to_subsample)
    return np.random.choice(np.arange(len(dataset.targets)), size=num, replace=False)

# Split a labelled training dataset into train and validation indices
def get_train_val_indices(train_dataset, val_instances_per_class=5):
    # Unique class labels in the dataset
    train_classes = list(set(train_dataset.targets))
    train_idxs = []
    val_idxs = []
    for cls in train_classes:
        # Indices of all samples belonging to the current class
        cls_idxs = np.where(np.array(train_dataset.targets) == cls)[0]

        # --- Minimum sample check per class ---
        if len(cls_idxs) < val_instances_per_class:
            raise ValueError(
                f"Class {cls} has only {len(cls_idxs)} samples, "
                f"which is fewer than val_instances_per_class={val_instances_per_class}."
            )
        
        # Randomly select validation instances for this class
        v_ = np.random.choice(cls_idxs, replace=False, size=val_instances_per_class)

        # Remaining instances are used for training
        t_ = [x for x in cls_idxs if x not in v_]

        train_idxs.extend(t_)
        val_idxs.extend(v_)

    return train_idxs, val_idxs

def get_owdfa_datasets(train_transform, test_transform, dataset_root,
                       train_classes, prop_train_labels=0.75, seed=2025, split_train_val=False,
                       crop_face=True, predictor_path=None, known_classes=None):
    """
    Construct datasets for the open-world setting.

    This function builds the following datasets:
      - train_labelled:
            Labelled training data sampled from known classes only.
            Optionally further split into train/validation subsets.
      - train_unlabelled:
            Unlabelled training data consisting of:
              (1) remaining samples from known classes, and
              (2) all samples from unknown (novel) classes.
      - val (optional):
            Validation set sampled from labelled training data.
      - test:
            Test set constructed from the same class pool as training,
            following the same class filtering rules.

    The final label space is unified across labelled, unlabelled,
    and test datasets via a shared target_transform.
    """
    np.random.seed(seed)
    
    # ------------------------------------------------------------------
    # Step 0: Construct the full training dataset (before label/unlabel split)
    # ------------------------------------------------------------------
    train_dataset = OWDFADataset(root=dataset_root,
                                 transform=train_transform, 
                                 train=True,
                                 known_classes=known_classes,
                                 train_classes=train_classes,
                                 crop_face=crop_face, predictor_path=predictor_path, seed=seed)
    
    
    # ------------------------------------------------------------------
    # Step 1: Construct the labelled training dataset
    #   - Keep only samples from known classes
    #   - Randomly subsample a proportion of instances as labelled data
    # ------------------------------------------------------------------

    # Filter the training dataset to known classes only
    train_dataset_labelled = subsample_classes(deepcopy(train_dataset), include_classes=known_classes)
    
    # Randomly select a subset of instances as labelled samples
    subsample_indices = subsample_instances(train_dataset_labelled, prop_indices_to_subsample=prop_train_labels)
    train_dataset_labelled = subsample_dataset(train_dataset_labelled, subsample_indices)
    
    # ------------------------------------------------------------------
    # Step 2 (optional): Split labelled data into train / validation sets
    # ------------------------------------------------------------------
    if split_train_val:
        train_idxs, val_idxs = get_train_val_indices(train_dataset_labelled, val_instances_per_class=5)
        train_dataset_labelled_split = subsample_dataset(deepcopy(train_dataset_labelled), train_idxs)
        val_dataset_labelled_split = subsample_dataset(deepcopy(train_dataset_labelled), val_idxs)
        # Use test-time transforms for validation
        val_dataset_labelled_split.transform = test_transform
    else:
        train_dataset_labelled_split, val_dataset_labelled_split = None, None


    # ------------------------------------------------------------------
    # Step 3: Construct the unlabelled training dataset
    #   - Remove labelled instances from the full training dataset
    #   - Remaining samples (known + unknown classes) are treated as unlabelled
    # ------------------------------------------------------------------
    labelled_uq = set(train_dataset_labelled.uq_idxs)
    unlabelled_indices = np.array(list(set(train_dataset.uq_idxs) - labelled_uq))
    train_dataset_unlabelled = subsample_dataset(deepcopy(train_dataset), unlabelled_indices)

    # ------------------------------------------------------------------
    # Step 4: Construct the test dataset
    # ------------------------------------------------------------------
    test_dataset = OWDFADataset(root=dataset_root,
                                 transform=test_transform, 
                                 train=False,
                                 known_classes=known_classes,
                                 train_classes=train_classes,
                                 crop_face=crop_face, predictor_path=predictor_path, seed=seed)
    
    # ------------------------------------------------------------------
    # Step 5: Define a unified target_transform for open-world classification
    #   - Known and unknown classes are mapped into a single contiguous label space
    #   - The same mapping is applied to labelled, unlabelled, and test datasets
    # ------------------------------------------------------------------
    unlabelled_classes = list(set(train_dataset.targets) - set(known_classes))
    target_xform_dict = {k: i for i, k in enumerate(list(known_classes) + unlabelled_classes)}

    train_dataset_labelled.target_transform = lambda x: target_xform_dict[x]
    train_dataset_unlabelled.target_transform = lambda x: target_xform_dict[x]
    test_dataset.target_transform = lambda x: target_xform_dict[x]

    # Select final labelled / validation datasets based on configuration
    train_dataset_labelled = train_dataset_labelled_split if split_train_val else train_dataset_labelled
    val_dataset_labelled = val_dataset_labelled_split if split_train_val else None

    # ------------------------------------------------------------------
    # Step 6: Log dataset statistics for verification
    # ------------------------------------------------------------------
    all_datasets = {
        'train_labelled': train_dataset_labelled,
        'train_unlabelled': train_dataset_unlabelled,
        'val': val_dataset_labelled,
        'test': test_dataset,
    }

    logger.info("[Dataset Statistics]")
    dataset_stats(all_datasets, known_classes=known_classes, target_xform_dict=target_xform_dict)

    return all_datasets

if __name__ == '__main__':
    # Root directory
    dataset_root = "../OWDFA40-Benchmark/data"
    predictor_path = "../OWDFA40-Benchmark/shape_predictor_68_face_landmarks.dat"

    # training and testing transforms
    train_transform = None
    test_transform = None

    # Protocol 1
    logger.info("-------------------------------------------------------------Protocol 1-------------------------------------------------------------")
    known_classes = [0, 1, 3, 6, 8, 11, 14, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39]
    train_classes = list(range(0,41))
    datasets = get_owdfa_datasets(train_transform, test_transform,
                                    dataset_root=dataset_root,
                                    train_classes=train_classes,
                                    prop_train_labels=0.75, seed=2025, split_train_val=False,
                                    crop_face=True, predictor_path=predictor_path, known_classes=known_classes)

    # Protocol 2
    logger.info("-------------------------------------------------------------Protocol 2-------------------------------------------------------------")
    known_classes = [0, 1, 3, 6, 8, 11, 14, 21, 23, 25, 27, 29, 31]
    train_classes = list(range(0,41))
    datasets = get_owdfa_datasets(train_transform, test_transform,
                                    dataset_root=dataset_root,
                                    train_classes=train_classes,
                                    prop_train_labels=0.75, seed=2025, split_train_val=False,
                                    crop_face=True, predictor_path=predictor_path, known_classes=known_classes)

    # Protocol 3
    logger.info("-------------------------------------------------------------Protocol 3-------------------------------------------------------------")
    known_classes = [0, 1, 3, 5, 6, 7, 8, 10, 11, 13, 14, 17, 18, 19, 20, 21, 23, 24, 25, 26, 27, 29, 31, 33, 35, 36, 37, 38, 39]
    train_classes = list(range(0,41))
    datasets = get_owdfa_datasets(train_transform, test_transform,
                                    dataset_root=dataset_root,
                                    train_classes=train_classes,
                                    prop_train_labels=0.75, seed=2025, split_train_val=False,
                                    crop_face=True, predictor_path=predictor_path, known_classes=known_classes)