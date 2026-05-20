"""Federated Data Loader for SHARD attack simulation.

Loads image datasets (MNIST, CIFAR-10) and generates reshuffled
mini-batches per epoch, simulating the federated learning data pipeline.
"""

import torch
import numpy as np
import torchvision
import torchvision.transforms as transforms


class FederatedDataLoader:
    """Loads image datasets and generates reshuffled mini-batches per epoch.

    Attributes:
        images: Flattened image tensor of shape (N, d).
        image_shape: Original image shape, e.g. (28, 28) or (32, 32, 3).
        n_samples: Number of samples N.
        batch_size: Batch size B.
        n_batches: Number of batches K = N // B.
    """

    def __init__(
        self,
        dataset: str,
        n_samples: int,
        batch_size: int,
        seed: int = 42,
        resize: int | None = None,
    ):
        """Initialize the data loader.

        Args:
            dataset: Dataset name, either "mnist" or "cifar10".
            n_samples: Number of samples N to use from the dataset.
            batch_size: Batch size B (must divide N evenly).
            seed: Random seed for reproducibility.
            resize: Optional target size to resize images to (e.g., 8 for 8x8).
                If None, uses original resolution.

        Raises:
            ValueError: If N is not divisible by B, or if dataset name
                is not supported.
        """
        # Validate dataset name
        supported = ("mnist", "cifar10", "tinyimagenet")
        if dataset.lower() not in supported:
            raise ValueError(
                f"Unsupported dataset '{dataset}'. "
                f"Supported datasets: {supported}"
            )

        # Validate divisibility
        if n_samples % batch_size != 0:
            raise ValueError(
                f"n_samples ({n_samples}) must be evenly divisible by "
                f"batch_size ({batch_size}), but "
                f"{n_samples} % {batch_size} = {n_samples % batch_size}"
            )

        self.n_samples = n_samples
        self.batch_size = batch_size
        self.n_batches = n_samples // batch_size
        self._rng = np.random.default_rng(seed)

        # Load dataset
        self.images, self.image_shape = self._load_dataset(
            dataset.lower(), n_samples, resize
        )

    def _load_dataset(
        self, dataset: str, n_samples: int, resize: int | None = None,
    ) -> tuple[torch.Tensor, tuple]:
        """Load the dataset from the local data folder (no download).

        Args:
            dataset: "mnist", "cifar10", or "tinyimagenet".
            n_samples: Number of samples to select.
            resize: Optional target size (e.g., 8 for 8x8).

        Returns:
            Tuple of (flattened images tensor of shape (N, d), original image shape).
        """
        xforms = []
        if resize is not None:
            xforms.append(transforms.Resize((resize, resize),
                                            antialias=True))
        xforms.append(transforms.ToTensor())
        transform = transforms.Compose(xforms)

        if dataset == "mnist":
            ds = torchvision.datasets.MNIST(
                root="./data", train=True, download=False, transform=transform
            )
            sz = resize or 28
            image_shape = (sz, sz)
        elif dataset == "cifar10":
            ds = torchvision.datasets.CIFAR10(
                root="./data", train=True, download=False, transform=transform
            )
            sz = resize or 32
            image_shape = (sz, sz, 3)
        else:  # tinyimagenet
            # Try loading from parquet file first (preferred)
            import os
            parquet_path = "./data/tinyimagenet.parquet"
            
            if os.path.exists(parquet_path):
                # Load from parquet file
                import pandas as pd
                from PIL import Image
                import io
                
                print(f"Loading TinyImageNet from {parquet_path}")
                df = pd.read_parquet(parquet_path)
                
                # Load images from bytes
                images = []
                for i in range(min(n_samples, len(df))):
                    img_bytes = df.iloc[i]['image']['bytes']
                    img_pil = Image.open(io.BytesIO(img_bytes)).convert('RGB')
                    
                    # Apply transforms
                    if resize is not None:
                        img_pil = img_pil.resize((resize, resize), Image.BILINEAR)
                    
                    # Convert to tensor and flatten
                    img_tensor = transforms.ToTensor()(img_pil)
                    images.append(img_tensor.flatten())
                
                images = torch.stack(images)
                sz = resize or 64
                image_shape = (sz, sz, 3)
                print(f"Loaded {len(images)} TinyImageNet images from parquet")
                return images, image_shape
            
            # Fallback: Try ImageFolder format
            tiny_paths = [
                "./data/tiny-imagenet-200/train",
                "./data/tiny-imagenet-synthetic/train",
            ]
            
            ds = None
            for tiny_path in tiny_paths:
                if os.path.exists(tiny_path):
                    ds = torchvision.datasets.ImageFolder(
                        root=tiny_path, transform=transform
                    )
                    print(f"Loaded TinyImageNet from {tiny_path}")
                    break
            
            if ds is None:
                # Generate synthetic TinyImageNet-like samples (64x64x3)
                print(f"TinyImageNet not found, using random synthetic samples")
                print(f"Run 'python experiments/exp5ii_generate_samples.py' for structured samples")
                sz = resize or 64
                image_shape = (sz, sz, 3)
                images = torch.rand(n_samples, sz * sz * 3)
                return images, image_shape
            
            sz = resize or 64
            image_shape = (sz, sz, 3)

        # Select first N samples and flatten to 1D vectors
        images = []
        for i in range(min(n_samples, len(ds))):
            img, _ = ds[i]  # img is a tensor from ToTensor()
            images.append(img.flatten())

        images = torch.stack(images)  # shape (N, d)
        return images, image_shape

    def get_epoch_batches(self) -> list[list[int]]:
        """Generate a random partition of [0, N-1] into K batches of size B.

        Returns:
            List of K lists, each containing B sample indices.
            The union of all lists equals {0, 1, ..., N-1}.
        """
        indices = self._rng.permutation(self.n_samples)
        batches = []
        for k in range(self.n_batches):
            start = k * self.batch_size
            end = start + self.batch_size
            batches.append(indices[start:end].tolist())
        return batches
