"""
KITTI Depth Dataset Loader
Supports:
1. val_selection_cropped (1000 images with RGB + depth) - for evaluation
2. Full train/val splits (requires KITTI Raw Data for RGB images)
"""
import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
from PIL import Image
import torchvision.transforms as transforms


class KITTIDepthDataset(Dataset):
    """
    KITTI Depth Dataset
    
    Data structure (depth_selection/val_selection_cropped):
    - image/: RGB images (1216x352)
    - groundtruth_depth/: GT depth maps (uint16 PNG, depth = value/256.0)
    - velodyne_raw/: Sparse LiDAR depth (optional)
    
    For full train/val splits, RGB images need to be downloaded from KITTI Raw Data.
    """
    
    def __init__(self, 
                 data_path,
                 split='val',
                 input_size=518,
                 min_depth=0.001,
                 max_depth=80.0,  # KITTI outdoor: 0-80m (vs NYU indoor: 0-10m)
                 augmentation=False,
                 use_cropped_val=True,
                 sample_ratio=None):
        """
        Args:
            data_path: Path to KITTI_depth_completion directory
            split: 'train', 'val', or 'test'
            input_size: Target image size (default: 518 for DINOv3)
            min_depth: Minimum depth value (meters)
            max_depth: Maximum depth value (meters), KITTI uses 80m
            augmentation: Whether to apply data augmentation
            use_cropped_val: Use val_selection_cropped (recommended for evaluation)
            sample_ratio: Use only a fraction of the dataset
        """
        super().__init__()
        
        self.data_path = Path(data_path)
        self.split = split
        self.input_size = input_size
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.augmentation = augmentation and (split == 'train')
        self.sample_ratio = sample_ratio
        
        # Try to find the dataset
        self._init_dataset(use_cropped_val)
        
        # Define transforms
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        
        # Resize transforms
        if self.augmentation:
            self.img_transform = transforms.Compose([
                transforms.Resize((input_size, input_size)),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            ])
        else:
            self.img_transform = transforms.Resize((input_size, input_size))
        
        self.depth_transform = transforms.Resize(
            (input_size, input_size),
            interpolation=transforms.InterpolationMode.NEAREST
        )
    
    def _init_dataset(self, use_cropped_val):
        """Initialize dataset paths and sample list"""
        self.samples = []
        
        # Check for different data formats
        # 1. val_selection_cropped (recommended for evaluation)
        cropped_val_path = self.data_path / 'depth_selection' / 'val_selection_cropped'
        
        # 2. Full train/val splits
        train_path = self.data_path / 'train'
        val_path = self.data_path / 'val'
        
        if use_cropped_val and cropped_val_path.exists() and self.split in ['val', 'test']:
            self._init_cropped_val(cropped_val_path)
        elif self.split == 'train' and train_path.exists():
            self._init_full_split(train_path, 'train')
        elif self.split == 'val' and val_path.exists():
            self._init_full_split(val_path, 'val')
        else:
            # Fallback: try cropped val for any split
            if cropped_val_path.exists():
                print(f"⚠️ Using val_selection_cropped for '{self.split}' split")
                self._init_cropped_val(cropped_val_path)
            else:
                raise FileNotFoundError(
                    f"KITTI Depth dataset not found at {self.data_path}\n"
                    f"Expected: {cropped_val_path} or {train_path}"
                )
        
        # Apply sample ratio
        if self.sample_ratio is not None and 0.0 < self.sample_ratio < 1.0:
            num_samples = int(len(self.samples) * self.sample_ratio)
            num_samples = max(1, num_samples)
            self.samples = self.samples[:num_samples]
            print(f"Using {self.sample_ratio*100:.1f}% of data: {num_samples} samples")
        
        print(f"KITTI Depth - Split: {self.split}, Samples: {len(self.samples)}")
    
    def _init_cropped_val(self, cropped_path):
        """Initialize from val_selection_cropped (has RGB + depth)"""
        image_dir = cropped_path / 'image'
        depth_dir = cropped_path / 'groundtruth_depth'
        
        if not image_dir.exists() or not depth_dir.exists():
            raise FileNotFoundError(f"Missing image or depth directory in {cropped_path}")
        
        # Get all image files
        image_files = sorted(image_dir.glob('*.png'))
        
        for img_path in image_files:
            # KITTI val_selection_cropped naming convention:
            # image: 2011_09_26_drive_0002_sync_image_0000000005_image_02.png
            # depth: 2011_09_26_drive_0002_sync_groundtruth_depth_0000000005_image_02.png
            # Need to replace '_image_' with '_groundtruth_depth_' in filename
            depth_name = img_path.name.replace('_image_', '_groundtruth_depth_', 1)
            depth_path = depth_dir / depth_name
            
            if depth_path.exists():
                self.samples.append({
                    'rgb_path': str(img_path),
                    'depth_path': str(depth_path),
                    'format': 'cropped_val'
                })
        
        print(f"Loaded {len(self.samples)} samples from val_selection_cropped")
    
    def _init_full_split(self, split_path, split_name):
        """
        Initialize from full train/val split
        Note: This requires KITTI Raw Data for RGB images
        """
        # Check if we have RGB images (from KITTI Raw)
        raw_data_path = self.data_path.parent / 'KITTI_Raw'
        
        drives = sorted(split_path.glob('*_drive_*_sync'))
        
        rgb_available = raw_data_path.exists()
        if not rgb_available:
            print(f"⚠️ KITTI Raw Data not found at {raw_data_path}")
            print(f"⚠️ RGB images are required for training. Please download KITTI Raw Data.")
            print(f"⚠️ Falling back to val_selection_cropped for evaluation only.")
            
            # Fall back to cropped val
            cropped_path = self.data_path / 'depth_selection' / 'val_selection_cropped'
            if cropped_path.exists():
                self._init_cropped_val(cropped_path)
            return
        
        for drive_path in drives:
            drive_name = drive_path.name
            date = '_'.join(drive_name.split('_')[:3])
            
            # Path to GT depth
            gt_path = drive_path / 'proj_depth' / 'groundtruth' / 'image_02'
            
            if not gt_path.exists():
                continue
            
            # Path to RGB images in KITTI Raw
            rgb_base = raw_data_path / date / drive_name / 'image_02' / 'data'
            
            for depth_file in sorted(gt_path.glob('*.png')):
                frame_id = depth_file.stem
                rgb_path = rgb_base / f'{frame_id}.png'
                
                if rgb_path.exists():
                    self.samples.append({
                        'rgb_path': str(rgb_path),
                        'depth_path': str(depth_file),
                        'format': 'full_split'
                    })
        
        print(f"Loaded {len(self.samples)} samples from {split_name} split")
    
    def __len__(self):
        return len(self.samples)
    
    def _load_depth(self, depth_path):
        """
        Load KITTI depth map
        Format: uint16 PNG, depth = pixel_value / 256.0 (meters)
        """
        depth = Image.open(depth_path)
        depth = np.array(depth, dtype=np.float32) / 256.0
        
        # Mark invalid pixels (value 0) as 0
        # Valid depth range for KITTI: typically 0-80m
        depth[depth <= 0] = 0
        
        return depth
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load RGB image
        image = Image.open(sample['rgb_path']).convert('RGB')
        
        # Load depth map
        depth = self._load_depth(sample['depth_path'])
        depth = Image.fromarray(depth)
        
        # Apply augmentation
        if self.augmentation:
            # Random horizontal flip (synchronized)
            if np.random.random() > 0.5:
                image = transforms.functional.hflip(image)
                depth = transforms.functional.hflip(depth)
            
            # Apply image-specific transforms
            image = self.img_transform(image)
        else:
            image = self.img_transform(image)
        
        # Resize depth
        depth = self.depth_transform(depth)
        
        # Convert to tensor
        image = self.to_tensor(image)
        image = self.normalize(image)
        
        depth = torch.from_numpy(np.array(depth)).float()
        
        # Clip depth values
        depth = torch.clamp(depth, self.min_depth, self.max_depth)
        
        return image, depth


def test_dataset():
    """Test the KITTI dataset loader"""
    import sys
    if len(sys.argv) < 2:
        print("Usage: python kitti_depth.py <data_path>")
        return
    
    kitti_path = sys.argv[1]
    print("=" * 60)
    print("Testing KITTI Depth Dataset Loader")
    print("=" * 60)
    
    try:
        dataset = KITTIDepthDataset(
            data_path=kitti_path,
            split='val',
            input_size=518,
            use_cropped_val=True
        )
        
        print(f"\nDataset loaded successfully!")
        print(f"  Size: {len(dataset)}")
        
        image, depth = dataset[0]
        print(f"  Image shape: {image.shape}, range: [{image.min():.3f}, {image.max():.3f}]")
        print(f"  Depth shape: {depth.shape}, range: [{depth.min():.3f}, {depth.max():.3f}]")
        
        valid_mask = depth > 0.001
        valid_ratio = valid_mask.float().mean().item() * 100
        print(f"  Valid depth ratio: {valid_ratio:.1f}%")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    
    print()


if __name__ == '__main__':
    test_dataset()

