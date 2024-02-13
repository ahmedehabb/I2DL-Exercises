import random
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

class CustomImageDataGenerator:
    def __init__(self, rotation_range=15, width_shift_range=0.1, height_shift_range=0.1,
                 horizontal_flip=True, brightness_range=(0.8, 1.2), zoom_range=0.1,
                 shear_range=0.2, noise_std=0.1):
        self.rotation_range = rotation_range
        self.width_shift_range = width_shift_range
        self.height_shift_range = height_shift_range
        self.horizontal_flip = horizontal_flip
        self.brightness_range = brightness_range
        self.zoom_range = zoom_range
        self.shear_range = shear_range
        self.noise_std = noise_std

    def random_rotation(self, image):
        angle = random.uniform(-self.rotation_range, self.rotation_range)
        return np.rot90(image, k=int(angle / 90))

    def random_shift(self, image):
        height, width = image.shape[:2]
        width_shift = int(random.uniform(-self.width_shift_range * width, self.width_shift_range * width))
        height_shift = int(random.uniform(-self.height_shift_range * height, self.height_shift_range * height))
        return np.roll(image, (height_shift, width_shift), axis=(0, 1))

    def horizontal_flip_image(self, image):
        if self.horizontal_flip and random.random() < 0.5:
            return np.fliplr(image)
        return image

    def random_brightness(self, image):
        factor = random.uniform(*self.brightness_range)
        return np.clip(image * factor, 0, 255).astype(np.uint8)

    def random_zoom(self, image):
        zoom_factor = random.uniform(1 - self.zoom_range, 1 + self.zoom_range)
        h, w = image.shape[:2]
        new_h, new_w = int(h * zoom_factor), int(w * zoom_factor)

        # Ensure the image remains the same size as the original
        image_resized = np.array(Image.fromarray(image).resize((w, h)))
        return image_resized

    def random_shear(self, image):
        shear_factor = random.uniform(-self.shear_range, self.shear_range)
        pil_img = Image.fromarray(image)
        transformed_img = pil_img.transform(
            image.shape[:2], Image.AFFINE, (1, shear_factor, 0, 0, 1, 0)
        )
        return np.array(transformed_img)

    def add_gaussian_noise(self, image):
        noise = np.random.normal(0, self.noise_std, image.shape)
        noisy_image = image + noise
        return np.clip(noisy_image, 0, 255).astype(np.uint8)

    def random_contrast(self, image):
        enhancer = ImageEnhance.Contrast(Image.fromarray(image))
        factor = random.uniform(0.5, 1.5)  # Adjust the contrast between 0.5 and 1.5
        return np.array(enhancer.enhance(factor))

    def random_blur(self, image):
        image_pil = Image.fromarray(image)
        radius = random.uniform(0.1, 2.0)  # Random blur radius
        return np.array(image_pil.filter(ImageFilter.GaussianBlur(radius)))

    def random_color(self, image):        
        enhancer = ImageEnhance.Color(Image.fromarray(image))
        factor = random.uniform(0.5, 1.5)  # Adjust the color balance
        return np.array(enhancer.enhance(factor))

    def random_cutout(self, image):
        mask_size = random.randint(4, 10)  # Random size of the cutout square
        h, w, _ = image.shape
        mask = np.ones((h, w, 3), np.uint8)
        y = np.random.randint(h)
        x = np.random.randint(w)
        y1 = np.clip(y - mask_size // 2, 0, h)
        y2 = np.clip(y + mask_size // 2, 0, h)
        x1 = np.clip(x - mask_size // 2, 0, w)
        x2 = np.clip(x + mask_size // 2, 0, w)
        mask[y1:y2, x1:x2] = 0
        image = image * mask
        return image
    
    # def apply_augmentation(self, image):
    #     image = self.random_rotation(image)
    #     image = self.random_shift(image)
    #     image = self.horizontal_flip_image(image)
    #     image = self.random_brightness(image)
    #     image = self.random_zoom(image)
    #     image = self.random_shear(image)
    #     image = self.add_gaussian_noise(image)
    #     return image
    
    def apply_augmentation(self, image):
        
        # Convert the image to the appropriate data type (uint8) and shape (H x W x C)
        image = image.astype(np.uint8)
        
        # Apply a random combination of augmentation techniques
        selected_augmentations = [
            self.random_rotation, 
            self.random_shift, 
            self.horizontal_flip_image, 
            self.random_brightness, 
            self.random_zoom, 
            self.random_shear, 
            self.add_gaussian_noise,
            self.random_contrast,
            self.random_blur,
            self.random_color,
            self.random_cutout
        ]

        # Randomly select a subset of augmentations for each image
        num_augmentations = 7  # Number of augmentations to apply to each image
        selected_augmentations_1 = random.sample(selected_augmentations, num_augmentations)
        selected_augmentations_2 = random.sample(selected_augmentations, num_augmentations)
        selected_augmentations_3 = random.sample(selected_augmentations, num_augmentations)

        random.shuffle(selected_augmentations_1)
        random.shuffle(selected_augmentations_2)
        # Apply selected augmentations to create two images
        image1 = image.copy()
        for augmentation in selected_augmentations_1:
            image1 = augmentation(image1)

        image2 = image.copy()
        for augmentation in selected_augmentations_2:
            image2 = augmentation(image2)

        return image1, image2


# Example usage:
# Assuming x_train is a list/array of PIL Image objects
# You can apply augmentation to each image in x_train using the CustomImageDataGenerator

# custom_generator = CustomImageDataGenerator()
# augmented_images = [(custom_generator.apply_augmentation(image_dict['image']),image_dict['label']) for image_dict in dataloaders["train"]]