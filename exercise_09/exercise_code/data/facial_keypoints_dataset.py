"""Dataset for facial keypoint detection"""

import os

import pandas as pd
import numpy as np
import torch

from .base_dataset import BaseDataset


class FacialKeypointsDataset(BaseDataset):
    """Dataset for facial keypoint detection"""
    def __init__(self, *args, train=True, transform=None, **kwargs):
        super().__init__(*args, **kwargs)
        file_name = "training.csv" if train else "val.csv"
        csv_file = os.path.join(self.root_path, file_name)
        val_csv = os.path.join(self.root_path, "val.csv")
        
        self.train = train

        if self.train:
            df = []
            for file in [csv_file, val_csv]:
              df.append(pd.read_csv(file))
            # Concatenate the DataFrames
            self.key_pts_frame = pd.concat(df, ignore_index=True)
        else:
            self.key_pts_frame = pd.read_csv(csv_file)
        self.key_pts_frame.dropna(inplace=True)
        self.key_pts_frame.reset_index(drop=True, inplace=True)
        self.transform = transform
        self.num_samples = len(self.key_pts_frame)
        self.flip_indices = None

        if self.train:
            # Generate indices for horizontally flipped images (only for training)
            self.flip_indices = np.arange(self.num_samples, 2 * self.num_samples)

    @staticmethod
    def _get_image(idx, key_pts_frame):
        img_str = key_pts_frame.loc[idx]['Image']
        img = np.array([
            int(item) for item in img_str.split()
        ]).reshape((96, 96))
        return np.expand_dims(img, axis=2).astype(np.uint8)

    @staticmethod
    def _get_keypoints(idx, key_pts_frame, shape=(15, 2)):
        keypoint_cols = list(key_pts_frame.columns)[:-1]
        key_pts = key_pts_frame.iloc[idx][keypoint_cols].values.reshape(shape)
        key_pts = (key_pts.astype(np.float) - 48.0) / 48.0
        return torch.from_numpy(key_pts).float()

    def _horizontal_flip(self, image, keypoints):
        flipped_image = np.flip(image, axis=1).copy()
        flipped_keypoints = keypoints.clone()
        flipped_keypoints[:, 0] = -keypoints[:, 0]  # Negate x-coordinate for flipped keypoints
        return flipped_image, flipped_keypoints

    def __len__(self):
        return 2 * self.num_samples if self.train else self.num_samples


    def __getitem__(self, idx):
        image = self._get_image(idx, self.key_pts_frame)
        keypoints = self._get_keypoints(idx, self.key_pts_frame)
        if self.transform:
            image = self.transform(image)
        return {'image': image, 'keypoints': keypoints}


    def __getitem__(self, idx):
        if self.train and idx >= self.num_samples:
            original_idx = idx - self.num_samples
            image = self._get_image(original_idx, self.key_pts_frame)
            keypoints = self._get_keypoints(original_idx, self.key_pts_frame)
            flipped_image, flipped_keypoints = self._horizontal_flip(image, keypoints)

            if self.transform:
                flipped_image = self.transform(flipped_image)

            return {'image': flipped_image, 'keypoints': flipped_keypoints}
        
        original_idx = idx
        image = self._get_image(original_idx, self.key_pts_frame)
        keypoints = self._get_keypoints(original_idx, self.key_pts_frame)

        if self.transform:
            image = self.transform(image)

        return {'image': image, 'keypoints': keypoints}