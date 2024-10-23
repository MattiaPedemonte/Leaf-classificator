# CNN for classification on Grapevine leaves
import tensorflow as tf                                    # TensorFlow deep learning framework
from tensorflow.keras import layers                   # Image loading and manipulation library
from tensorflow.keras.models import Sequential, Model      # Sequential and Functional API for building models
from tensorflow.keras.optimizers import Adam               # Adam optimizer for model training
from tensorflow.keras.callbacks import EarlyStopping       # Early stopping callback for model training
from tensorflow.keras.regularizers import l1, l2           # L1 and L2 regularization for model regularization
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # Data augmentation and preprocessing for images
from tensorflow.keras.layers import Dense, Flatten, Dropout, GlobalAveragePooling2D, AveragePooling2D, MaxPooling2D, BatchNormalization  
# Various types of layers for building neural networks
from tensorflow.keras.applications import DenseNet121, EfficientNetB4, Xception, VGG16, VGG19   # Pre-trained models for transfer learning


def GenerateModel(num_classes: int) -> Model:
    # Load the VGG19 model without the top layer (i.e., the fully connected layers)
    vgg19_model = tf.keras.applications.VGG19(
        include_top=False,        # Exclude the top layer
        weights='imagenet',      # Load pre-trained weights from ImageNet
        input_shape=(256, 256, 3) # Specify input shape
    )
    
    for layer in vgg19_model.layers[:-3]:  # Freeze all layers except the last 4
        layer.trainable = False
    
    x = vgg19_model.output
    x = MaxPooling2D()(x)
    x = Flatten()(x)
    x = Dense(64, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=vgg19_model.input, outputs=predictions)

    # Check the model summary
    model.summary()

    return model