from torchvision import transforms as T


def get_train_transforms(image_size: tuple[int, int] = (224, 224)):
    """Training-time image augmentation."""
    return T.Compose([
        T.Resize(image_size),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        T.RandomAffine(degrees=5, translate=(0.05, 0.05)),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])


def get_eval_transforms(image_size: tuple[int, int] = (224, 224)):
    """Evaluation-time image preprocessing (no augmentation)."""
    return T.Compose([
        T.Resize(image_size),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
