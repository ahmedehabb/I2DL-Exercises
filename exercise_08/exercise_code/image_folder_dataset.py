"""
Definition of ImageFolderDataset dataset class
"""

# pylint: disable=too-few-public-methods

import os
import torch
from .base_dataset import Dataset
import random
from torchvision import transforms


class ImageFolderDataset(Dataset):
    """CIFAR-10 dataset class"""

    def __init__(self, *args,
                 root=None,
                 images=None,
                 labels=None,
                 transform=None,
                 download_url="https://i2dl.vc.in.tum.de/static/data/mnist.zip",
                 **kwargs):
        super().__init__(*args,
                         download_url=download_url,
                         root=root,
                         **kwargs)

        print(download_url)
        self.mode = images
        self.images = torch.load(os.path.join(root, images))
        if labels is not None:
            self.labels = torch.load(os.path.join(root, labels))
        else:
            self.labels = None

        if self.mode == "train_images.pt":
            self.images = torch.cat((self.images, torch.load(os.path.join(root, "test_images.pt"))), dim=0)
            if self.labels is not None:
                self.labels = torch.cat((self.labels, torch.load(os.path.join(root, "test_labels.pt"))), dim=0)
        
        self.transformations = [
            transforms.RandomRotation(degrees=(-15, 15)),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), shear=(-10, 10)),
            transforms.RandomPerspective(distortion_scale=0.2),
            transforms.RandomResizedCrop(size=(28, 28), scale=(0.9, 1.1)),
            transforms.PILToTensor(),
        ]
        # self.transformations = [
        #     transforms.RandomHorizontalFlip(),  # Randomly flip images horizontally
        #     transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  # Adjust color brightness, contrast, saturation
        #     transforms.RandomRotation(15),  # Randomly rotate images within [-15, 15] degrees
        #     transforms.ToTensor(),  # Convert images to PyTorch tensors
        # ]
        # self.transform = transform
      
        if self.mode == "train_images.pt":
            augmented_images = []
            augmented_labels = []
            for i in range(len(self.images)):
                for j in range(20):
                    pil_image = transforms.ToPILImage()(self.images[i])


                    # Compose the transformations (excluding normalization)
                    composed_transforms = transforms.Compose(self.transformations)

                    # Apply the composed transformations
                    augmented_img = composed_transforms(pil_image)

                    # tensor_img = transforms.ToTensor()(augmented_img)
                    
                    # Clip pixel values between 0 and 1
                    tensor_img = torch.clamp(augmented_img, 0, 1)
                    
                    augmented_images.append(tensor_img)
                    if self.labels is not None:
                        augmented_labels.append(self.labels[i])
            self.images = torch.cat((self.images, torch.stack(augmented_images)))
            if self.labels is not None:
                self.labels = torch.cat((self.labels, torch.tensor(augmented_labels)))
            else:
                self.labels = None

            if self.labels is not None:
                combined_data = list(zip(self.images, self.labels))
                random.shuffle(combined_data)
                self.images, self.labels = zip(*combined_data)
                self.images = torch.stack(self.images)
                self.labels = torch.stack(self.labels)
            else:
                # Convert tensor to a list, shuffle it, and then convert back to a tensor
                image_list = self.images.tolist()
                random.shuffle(image_list)
                self.images = torch.tensor(image_list)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):

        image = self.images[index]
        # if self.transform is not None and self.mode != "train_images.pt":
        #     image = self.transform(image)
        if self.labels is not None:
            return image, self.labels[index]
        else:
            return image


# """
# Definition of ImageFolderDataset dataset class
# """

# # pylint: disable=too-few-public-methods

# import os
# import torch
# from .base_dataset import Dataset


# class ImageFolderDataset(Dataset):
#     """CIFAR-10 dataset class"""

#     def __init__(self, *args,
#                  root=None,
#                  images=None,
#                  labels=None,
#                  transform=None,
#                  download_url="https://i2dl.vc.in.tum.de/static/data/mnist.zip",
#                  **kwargs):
#         super().__init__(*args,
#                          download_url=download_url,
#                          root=root,
#                          **kwargs)
#         print(download_url)
#         self.images = torch.load(os.path.join(root, images))
#         if labels is not None:
#             self.labels = torch.load(os.path.join(root, labels))
#         else:
#             self.labels = None
#         # transform function that we will apply later for data preprocessing
#         self.transform = transform

#     def __len__(self):
#         return len(self.images)

#     def __getitem__(self, index):

#         image = self.images[index]
#         if self.transform is not None:
#             image = self.transform(image)
#         if self.labels is not None:
#             return image, self.labels[index]
#         else:
#             return image
