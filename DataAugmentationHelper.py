import numpy as np
import pandas as pd

import math
import cv2
from tensorflow.keras.utils import Sequence
import tensorflow as tf
import os

def apply_random_rotation(image, bbox, max_angle=15):
    angle = np.random.uniform(-max_angle, max_angle)
    rad_angle = math.radians(angle)
    
    image_center = (image.shape[1] / 2, image.shape[0] / 2)
    rotation_matrix = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]))
    
    x_min, y_min, x_max, y_max = bbox
    box_points = np.array([[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]])
    points_ones = np.hstack([box_points, np.ones((4, 1))])
    rotated_points = rotation_matrix.dot(points_ones.T).T
    
    x_min_rot, y_min_rot = rotated_points[:, 0].min(), rotated_points[:, 1].min()
    x_max_rot, y_max_rot = rotated_points[:, 0].max(), rotated_points[:, 1].max()
    rotated_bbox = [x_min_rot, y_min_rot, x_max_rot, y_max_rot]
    
    return rotated_image, rotated_bbox

def apply_random_shearing(image, bbox, max_shear=15):
    shear_x = np.random.uniform(-math.radians(max_shear), math.radians(max_shear))
    shear_y = np.random.uniform(-math.radians(max_shear), math.radians(max_shear))
    shear_matrix = np.array([[1, math.tan(shear_x), 0], [math.tan(shear_y), 1, 0]], dtype=np.float32)
    sheared_image = cv2.warpAffine(image, shear_matrix, (image.shape[1], image.shape[0]))
    
    x_min, y_min, x_max, y_max = bbox
    box_points = np.array([[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]])
    points_ones = np.hstack([box_points, np.ones((4, 1))])
    sheared_points = shear_matrix.dot(points_ones.T).T
    
    x_min_shear, y_min_shear = sheared_points[:, 0].min(), sheared_points[:, 1].min()
    x_max_shear, y_max_shear = sheared_points[:, 0].max(), sheared_points[:, 1].max()
    sheared_bbox = [x_min_shear, y_min_shear, x_max_shear, y_max_shear]
    
    return sheared_image, sheared_bbox

class DataGenerator(Sequence):
    def __init__(self, dataframe, image_folder, output_folder, batch_size=16, target_size=(512, 512), 
                 augment=True, augmentations_per_image=2, shuffle=True):
        """
        DataGenerator with optional data augmentation to expand the dataset.
        
        Parameters:
        - dataframe: Original DataFrame with image paths and bbox details.
        - image_folder: Folder containing the original images.
        - output_folder: Folder to save augmented images.
        - batch_size: Number of samples per batch.
        - target_size: Tuple indicating image size (height, width).
        - augment: Whether to augment data.
        - augmentations_per_image: Number of augmented images to create per original image.
        - shuffle: Whether to shuffle data after each epoch.
        """
        self.dataframe = dataframe
        self.image_folder = image_folder
        self.output_folder = output_folder
        self.batch_size = batch_size
        self.target_size = target_size
        self.augment = augment
        self.augmentations_per_image = augmentations_per_image
        self.shuffle = shuffle

        if self.augment:
            self.dataframe = self.augment_dataset()
        
        self.on_epoch_end()
    
    def augment_dataset(self):
        augmented_data = []
        os.makedirs(self.output_folder, exist_ok=True)

        for index, row in self.dataframe.iterrows():
            image_path = os.path.join(self.image_folder, row['image_name'])
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image at {image_path}")
            
            original_height, original_width = image.shape[:2]

            # Original bounding box (absolute coordinates)
            bbox = [
                int(row['bbox_x']),
                int(row['bbox_y']),
                int(row['bbox_x'] + row['bbox_width']),
                int(row['bbox_y'] + row['bbox_height'])
            ]

            for i in range(self.augmentations_per_image):
                augmented_image = image.copy()
                augmented_bbox = bbox.copy()

                # Apply random rotation
                if np.random.rand() < 0.5:
                    augmented_image, augmented_bbox = apply_random_rotation(augmented_image, augmented_bbox)

                # Apply random shearing
                if np.random.rand() < 0.5:
                    augmented_image, augmented_bbox = apply_random_shearing(augmented_image, augmented_bbox)

                # Save augmented image
                augmented_image_name = f"{row['image_name'].split('.')[0]}_aug_{i}.jpg"
                augmented_image_path = os.path.join(self.output_folder, augmented_image_name)
                cv2.imwrite(augmented_image_path, augmented_image)

                # Normalize bounding box and update dataframe
                augmented_data.append({
                    'image_name': augmented_image_name,
                    'bbox_x': augmented_bbox[0],
                    'bbox_y': augmented_bbox[1],
                    'bbox_width': augmented_bbox[2] - augmented_bbox[0],
                    'bbox_height': augmented_bbox[3] - augmented_bbox[1],
                    'label_name': row['label_name']
                })

        augmented_dataframe = pd.concat([self.dataframe, pd.DataFrame(augmented_data)], ignore_index=True)
        return augmented_dataframe

    def __len__(self):
        return int(np.ceil(len(self.dataframe) / self.batch_size))
    
    def __getitem__(self, index):
        start_idx = index * self.batch_size
        end_idx = min(start_idx + self.batch_size, len(self.dataframe))
        batch_indices = self.indices[start_idx:end_idx]
        
        X, y = self.__data_generation(batch_indices)
        
        return X, y
    
    def on_epoch_end(self):
        self.indices = np.arange(len(self.dataframe))
        if self.shuffle:
            np.random.shuffle(self.indices)
    
    def __data_generation(self, batch_indices):
        batch_size = len(batch_indices)
        X = np.zeros((batch_size, *self.target_size, 3), dtype=np.float32)
        y_class = np.zeros((batch_size, 1))  
        y_bbox = np.zeros((batch_size, 4))
        
        for i, idx in enumerate(batch_indices):
            row = self.dataframe.iloc[idx]
            image_path = os.path.join(self.output_folder, row['image_name'])
            
            # Load and preprocess image
            image, bbox = self.preprocess_image(image_path, row)
            X[i] = image
        
            # Updated classification logic to handle multiple labels
            if row['label_name'] == 'empty-shelf':
                y_class[i, 0] = 1  # Empty shelf
            elif row['label_name'] == 'product':
                y_class[i, 0] = 2  # Product
            else:
                y_class[i, 0] = 0  # Other/background
            
            # Ensure bounding box has 4 coordinates
            if bbox is None:
                y_bbox[i] = [0, 0, 0, 0]  # Or use a default bbox if none is present
            else:
                y_bbox[i] = bbox
        
        return X, {'classification_output': y_class, 'bbox_output': y_bbox}
        
    def preprocess_image(self, image_path, row):
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image at {image_path}")
            
        original_height, original_width = image.shape[:2]
        
        # Normalize bbox coordinates
        x_min = row['bbox_x'] / original_width
        y_min = row['bbox_y'] / original_height
        x_max = (row['bbox_x'] + row['bbox_width']) / original_width
        y_max = (row['bbox_y'] + row['bbox_height']) / original_height
        bbox = [x_min, y_min, x_max, y_max]
        
        # Resize and normalize image
        image = cv2.resize(image, self.target_size)
        image = image.astype(np.float32) / 255.0
        
        return image, np.array(bbox)
