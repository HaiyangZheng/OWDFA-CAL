from torchvision import transforms as T
import torchvision.transforms.functional as F
import random

# ------------------------------------------------------------
# View generators for contrastive / semi-supervised learning
# ------------------------------------------------------------
class ContrastiveLearningViewGenerator:
    """
    Generate multiple augmented views of the same image for
    contrastive learning.

    If `base_transform` is a list, each transform in the list
    is applied once.
    Otherwise, the same transform is applied `n_views` times.
    """
    def __init__(self, base_transform, n_views: int = 2):
        self.base_transform = base_transform
        self.n_views = n_views

    def __call__(self, img):
        if isinstance(self.base_transform, list):
            return [t(img) for t in self.base_transform]
        else:
            return [self.base_transform(img) for _ in range(self.n_views)]

class WS_ViewGenerator:
    """
    Generate a weak–strong pair of augmented views from the same image.

    This is commonly used in semi-supervised learning, where:
      - the weak view is used for pseudo-label generation
      - the strong view is used for consistency regularization
    """
    def __init__(self, weak_transform, strong_transform):
        self.weak_transform   = weak_transform
        self.strong_transform = strong_transform
    def __call__(self, img):
        if isinstance(self.weak_transform, list):
            return [self.weak_transform(img), self.strong_transform(img)]
        else:
            return [self.weak_transform(img), self.strong_transform(img)]
        

# ------------------------------------------------------------
# Helper augmentation primitives
# ------------------------------------------------------------
def _maybe_random_resized_crop(img, size_hw, scale=(0.5, 1.0), p=0.2):
    """
    With probability `p`, apply RandomResizedCrop;
    otherwise, directly resize the image to the target size.
    """
    if random.random() < p:
        return T.RandomResizedCrop(size_hw, scale=scale)(img)
    return F.resize(img, size_hw)

def _maybe_brightness_contrast(p=0.2, brightness=0.4, contrast=0.4):
    """
    With probability `p`, apply ColorJitter on brightness and contrast.
    """
    return T.RandomApply(
        [T.ColorJitter(brightness=brightness, contrast=contrast)],
        p=p
    )


# ------------------------------------------------------------
# Main transform factory (weak / strong / test)
# ------------------------------------------------------------
def create_data_transforms_ws(args, split='train'):
    image_size = getattr(args, 'image_size', 224)
    mean       = getattr(args, 'mean',  [0.485, 0.456, 0.406])
    std        = getattr(args, 'std',   [0.229, 0.224, 0.225])

    size_hw = (image_size, image_size)        # (H, W)
    crop_hw = int((256 / 224) * image_size)

    if split == 'train':
        # ----------------------------------------------------
        # Strong view
        # ----------------------------------------------------
        strong_transform = T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            lambda img: _maybe_random_resized_crop(img, size_hw=crop_hw,
                                                  scale=(0.5, 1.0), p=0.2),
            _maybe_brightness_contrast(p=0.2, brightness=0.4, contrast=0.4),
            T.Resize(size_hw, interpolation=T.InterpolationMode.BILINEAR,
                     antialias=True),
            T.ToTensor(),
            T.Normalize(mean, std),
        ])

        # ----------------------------------------------------
        # Weak view
        # ----------------------------------------------------
        weak_transform = T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.Resize(size_hw, interpolation=T.InterpolationMode.BILINEAR,
                     antialias=True),
            T.ToTensor(),
            T.Normalize(mean, std),
        ])

        return weak_transform, strong_transform

    # --------------------------------------------------------
    # Test / validation
    # --------------------------------------------------------
    test_transform = T.Compose([
        T.Resize(size_hw, interpolation=T.InterpolationMode.BILINEAR,
                 antialias=True),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])
    return test_transform