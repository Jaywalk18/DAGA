import torch
from torchvision import transforms as T
from torch.utils.data import Dataset
import os
import numpy as np
from PIL import Image


def get_segmentation_dataset(args):
    """
    Load and prepare segmentation dataset (ADE20K, Cityscapes, Pascal VOC, COCO)
    """
    if args.dataset == "ade20k":
        return get_ade20k_dataset(args)
    elif args.dataset == "cityscapes":
        return get_cityscapes_dataset(args)
    elif args.dataset == "voc" or args.dataset == "pascal_voc":
        return get_voc_dataset(args)
    elif args.dataset == "coco":
        return get_coco_seg_dataset(args)
    else:
        raise ValueError(f"Unknown segmentation dataset: {args.dataset}")


def get_ade20k_dataset(args):
    """Load ADE20K dataset for semantic segmentation"""
    
    train_img_dir = os.path.join(args.data_path, "images/ADE/training")
    val_img_dir = os.path.join(args.data_path, "images/ADE/validation")
    train_ann_dir = os.path.join(args.data_path, "images/ADE/training")  # Annotations are in same dir
    val_ann_dir = os.path.join(args.data_path, "images/ADE/validation")
    
    # Verify paths exist
    if not os.path.exists(train_img_dir):
        raise FileNotFoundError(f"ADE20K train images not found: {train_img_dir}")
    if not os.path.exists(val_img_dir):
        raise FileNotFoundError(f"ADE20K val images not found: {val_img_dir}")
    if not os.path.exists(train_ann_dir):
        raise FileNotFoundError(f"ADE20K train annotations not found: {train_ann_dir}")
    if not os.path.exists(val_ann_dir):
        raise FileNotFoundError(f"ADE20K val annotations not found: {val_ann_dir}")
    
    # Define transforms
    train_transform = T.Compose([
        T.Resize((args.input_size, args.input_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    val_transform = T.Compose([
        T.Resize((args.input_size, args.input_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    mask_transform = T.Compose([
        T.Resize((args.input_size, args.input_size), interpolation=T.InterpolationMode.NEAREST),
    ])
    
    # Load datasets
    train_dataset = ADE20KDataset(
        img_dir=train_img_dir,
        ann_dir=train_ann_dir,
        transform=train_transform,
        mask_transform=mask_transform
    )
    
    val_dataset = ADE20KDataset(
        img_dir=val_img_dir,
        ann_dir=val_ann_dir,
        transform=val_transform,
        mask_transform=mask_transform
    )
    
    # ADE20K has 150 classes + 1 background
    num_classes = 150
    
    return train_dataset, val_dataset, num_classes


def get_coco_seg_dataset(args):
    """Load COCO dataset for semantic segmentation"""
    raise NotImplementedError("COCO segmentation dataset not implemented yet")


def get_voc_dataset(args):
    """Load Pascal VOC 2012 dataset for semantic segmentation"""
    
    # VOC directory structure:
    # data_path/VOCdevkit/VOC2012/JPEGImages/*.jpg
    # data_path/VOCdevkit/VOC2012/SegmentationClass/*.png
    # data_path/VOCdevkit/VOC2012/ImageSets/Segmentation/{train,val,trainval}.txt
    
    voc_root = os.path.join(args.data_path, "VOCdevkit/VOC2012")
    img_dir = os.path.join(voc_root, "JPEGImages")
    ann_dir = os.path.join(voc_root, "SegmentationClass")
    split_dir = os.path.join(voc_root, "ImageSets/Segmentation")
    
    # Verify paths exist
    if not os.path.exists(img_dir):
        raise FileNotFoundError(f"Pascal VOC images not found: {img_dir}")
    if not os.path.exists(ann_dir):
        raise FileNotFoundError(f"Pascal VOC annotations not found: {ann_dir}")
    
    # Define transforms
    train_transform = T.Compose([
        T.Resize((args.input_size, args.input_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    val_transform = T.Compose([
        T.Resize((args.input_size, args.input_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    mask_transform = T.Compose([
        T.Resize((args.input_size, args.input_size), interpolation=T.InterpolationMode.NEAREST),
    ])
    
    # Load datasets
    train_dataset = PascalVOCDataset(
        img_dir=img_dir,
        ann_dir=ann_dir,
        split_file=os.path.join(split_dir, "train.txt"),
        transform=train_transform,
        mask_transform=mask_transform
    )
    
    val_dataset = PascalVOCDataset(
        img_dir=img_dir,
        ann_dir=ann_dir,
        split_file=os.path.join(split_dir, "val.txt"),
        transform=val_transform,
        mask_transform=mask_transform
    )
    
    # Pascal VOC has 21 classes (20 object classes + 1 background)
    # But we treat background as class 0, so num_classes = 21
    num_classes = 21
    
    return train_dataset, val_dataset, num_classes


def get_cityscapes_dataset(args):
    """Load Cityscapes dataset for semantic segmentation"""
    
    # Cityscapes directory structure:
    # data_path/leftImg8bit/{train,val}/{city}/*_leftImg8bit.png
    # data_path/gtFine/{train,val}/{city}/*_gtFine_labelIds.png
    
    train_img_dir = os.path.join(args.data_path, "leftImg8bit/train")
    val_img_dir = os.path.join(args.data_path, "leftImg8bit/val")
    train_ann_dir = os.path.join(args.data_path, "gtFine/train")
    val_ann_dir = os.path.join(args.data_path, "gtFine/val")
    
    # Verify paths exist
    if not os.path.exists(train_img_dir):
        raise FileNotFoundError(f"Cityscapes train images not found: {train_img_dir}")
    if not os.path.exists(val_img_dir):
        raise FileNotFoundError(f"Cityscapes val images not found: {val_img_dir}")
    
    # Define transforms
    train_transform = T.Compose([
        T.Resize((args.input_size, args.input_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    val_transform = T.Compose([
        T.Resize((args.input_size, args.input_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    mask_transform = T.Compose([
        T.Resize((args.input_size, args.input_size), interpolation=T.InterpolationMode.NEAREST),
    ])
    
    # Load datasets
    train_dataset = CityscapesDataset(
        img_dir=train_img_dir,
        ann_dir=train_ann_dir,
        transform=train_transform,
        mask_transform=mask_transform
    )
    
    val_dataset = CityscapesDataset(
        img_dir=val_img_dir,
        ann_dir=val_ann_dir,
        transform=val_transform,
        mask_transform=mask_transform
    )
    
    # Cityscapes has 19 training classes
    num_classes = 19
    
    return train_dataset, val_dataset, num_classes


class PascalVOCDataset(Dataset):
    """Pascal VOC 2012 Semantic Segmentation Dataset"""
    
    def __init__(self, img_dir, ann_dir, split_file, transform=None, mask_transform=None):
        self.img_dir = img_dir
        self.ann_dir = ann_dir
        self.transform = transform
        self.mask_transform = mask_transform
        
        # Load image IDs from split file
        self.samples = []
        with open(split_file, 'r') as f:
            for line in f:
                img_id = line.strip()
                if img_id:
                    img_path = os.path.join(img_dir, f"{img_id}.jpg")
                    ann_path = os.path.join(ann_dir, f"{img_id}.png")
                    if os.path.exists(img_path) and os.path.exists(ann_path):
                        self.samples.append((img_path, ann_path))
        
        print(f"Pascal VOC Dataset: {len(self.samples)} samples loaded from {split_file}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, ann_path = self.samples[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Load mask (palette PNG)
        mask_img = Image.open(ann_path)
        mask = np.array(mask_img, dtype=np.int64)
        
        # VOC uses 255 as the ignore/border class
        # Classes: 0 = background, 1-20 = object classes, 255 = ignore
        # Keep as is, just ensure 255 stays as ignore index
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        if self.mask_transform:
            mask_pil = Image.fromarray(mask.astype(np.uint8))
            mask_pil = self.mask_transform(mask_pil)
            mask = torch.from_numpy(np.array(mask_pil, dtype=np.int64))
        else:
            mask = torch.from_numpy(mask)
        
        return image, mask


class CityscapesDataset(Dataset):
    """Cityscapes Semantic Segmentation Dataset"""
    
    # Cityscapes label mapping: original labelIds -> trainIds (0-18)
    # See: https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py
    LABEL_MAPPING = {
        0: 255,   # unlabeled
        1: 255,   # ego vehicle
        2: 255,   # rectification border
        3: 255,   # out of roi
        4: 255,   # static
        5: 255,   # dynamic
        6: 255,   # ground
        7: 0,     # road
        8: 1,     # sidewalk
        9: 255,   # parking
        10: 255,  # rail track
        11: 2,    # building
        12: 3,    # wall
        13: 4,    # fence
        14: 255,  # guard rail
        15: 255,  # bridge
        16: 255,  # tunnel
        17: 5,    # pole
        18: 255,  # polegroup
        19: 6,    # traffic light
        20: 7,    # traffic sign
        21: 8,    # vegetation
        22: 9,    # terrain
        23: 10,   # sky
        24: 11,   # person
        25: 12,   # rider
        26: 13,   # car
        27: 14,   # truck
        28: 15,   # bus
        29: 255,  # caravan
        30: 255,  # trailer
        31: 16,   # train
        32: 17,   # motorcycle
        33: 18,   # bicycle
        -1: 255,  # license plate
    }
    
    def __init__(self, img_dir, ann_dir, transform=None, mask_transform=None):
        self.img_dir = img_dir
        self.ann_dir = ann_dir
        self.transform = transform
        self.mask_transform = mask_transform
        
        # Get all image files from city subdirectories
        self.samples = []
        for city in os.listdir(img_dir):
            city_img_dir = os.path.join(img_dir, city)
            city_ann_dir = os.path.join(ann_dir, city)
            if os.path.isdir(city_img_dir):
                for img_file in os.listdir(city_img_dir):
                    if img_file.endswith('_leftImg8bit.png'):
                        # Get corresponding annotation file
                        ann_file = img_file.replace('_leftImg8bit.png', '_gtFine_labelIds.png')
                        img_path = os.path.join(city_img_dir, img_file)
                        ann_path = os.path.join(city_ann_dir, ann_file)
                        if os.path.exists(ann_path):
                            self.samples.append((img_path, ann_path))
        
        print(f"Cityscapes Dataset: {len(self.samples)} samples loaded")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, ann_path = self.samples[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Load mask (labelIds)
        mask_img = Image.open(ann_path)
        mask = np.array(mask_img, dtype=np.int64)
        
        # Convert labelIds to trainIds using mapping
        mask_mapped = np.full_like(mask, 255)
        for label_id, train_id in self.LABEL_MAPPING.items():
            mask_mapped[mask == label_id] = train_id
        mask = mask_mapped
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        if self.mask_transform:
            mask_pil = Image.fromarray(mask.astype(np.uint8))
            mask_pil = self.mask_transform(mask_pil)
            mask = torch.from_numpy(np.array(mask_pil, dtype=np.int64))
        else:
            mask = torch.from_numpy(mask)
        
        return image, mask


class ADE20KDataset(Dataset):
    """ADE20K Semantic Segmentation Dataset"""
    def __init__(self, img_dir, ann_dir, transform=None, mask_transform=None):
        self.img_dir = img_dir
        self.ann_dir = ann_dir
        self.transform = transform
        self.mask_transform = mask_transform
        
        # Recursively get all image files from subdirectories
        self.img_files = []
        for root, dirs, files in os.walk(img_dir):
            for f in files:
                if f.endswith('.jpg'):
                    self.img_files.append(os.path.join(root, f))

    def __len__(self):
        return len(self.img_files)
    
    def _validate_samples(self, num_samples=3):
        """Validate a few samples to check mask value ranges"""
        print(f"Validating {num_samples} samples...")
        for i in range(num_samples):
            img_path = self.img_files[i]
            mask_base = img_path.replace('.jpg', '')
            mask_path = os.path.join(mask_base, f"{os.path.basename(mask_base)}_seg.png")
            if os.path.exists(mask_path):
                mask = np.array(Image.open(mask_path), dtype=np.int64)
                unique_vals = np.unique(mask)
                print(f"  Sample {i}: mask values in range [{unique_vals.min()}, {unique_vals.max()}]")
                if unique_vals.max() > 150:
                    print(f"    Warning: Found value {unique_vals.max()} > 150")
            else:
                print(f"  Sample {i}: mask not found at {mask_path}")

    def __getitem__(self, idx):
        # Image path is already absolute from os.walk
        img_path = self.img_files[idx]
        image = Image.open(img_path).convert('RGB')
        
        # Load mask (annotation) - _seg.png is in the same directory as .jpg, not in subfolder
        # Example: /path/to/ADE_val_00000749.jpg -> /path/to/ADE_val_00000749_seg.png
        mask_path = img_path.replace('.jpg', '_seg.png')
        
        if os.path.exists(mask_path):
            mask_img = Image.open(mask_path)
            # ADE20K masks are stored as RGB images, class index is in R channel
            mask_arr = np.array(mask_img, dtype=np.int64)
            if len(mask_arr.shape) == 3:  # RGB format
                mask = mask_arr[:, :, 0]  # Use R channel
            else:
                mask = mask_arr
            
            # ADE20K: 0 is background/unlabeled, 1-150 are object classes
            # Convert to: 255 is ignore (background), 0-149 are classes
            mask = mask.copy()
            mask[mask == 0] = 255  # Background becomes ignore index
            mask[mask != 255] -= 1  # Shift 1-150 to 0-149
        else:
            # If no mask found, create empty mask with ignore index
            print(f"Warning: Mask not found at {mask_path}")
            mask = np.full((image.size[1], image.size[0]), 255, dtype=np.int64)
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        if self.mask_transform:
            # Convert mask to PIL for transform, then back to tensor
            mask_pil = Image.fromarray(mask.astype(np.uint8))
            mask_pil = self.mask_transform(mask_pil)
            mask = torch.from_numpy(np.array(mask_pil, dtype=np.int64))
        else:
            mask = torch.from_numpy(mask)
        
        return image, mask
