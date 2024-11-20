from tensorflow.keras import layers, models, Input
import tensorflow as tf

def build_rcnn_model(input_shape):
    input_tensor = Input(shape=input_shape)
    
    # CNN backbone
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_tensor)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    x = layers.Flatten()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    
    # Classification head (now multi-class)
    classification_output = layers.Dense(3, activation='softmax', name='classification_output')(x)
    
    # Bounding box regression head
    bbox_output = layers.Dense(4, name='bbox_output')(x)
    
    model = tf.keras.Model(inputs=input_tensor, outputs=[bbox_output, classification_output])
    return model