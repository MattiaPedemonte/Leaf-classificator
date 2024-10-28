# CNN for classification on Grapevine leaves
from importLibrary import *

def GenerateVGG19Model(num_classes: int) -> Model:
    # Load the VGG19 model without the top layer (i.e., the fully connected layers)
    vgg19_model = tf.keras.applications.VGG19(
        include_top=False,        # Exclude the top layer
        weights='imagenet',      # Load pre-trained weights from ImageNet
        input_shape=(256, 256, 3) # Specify input shape
    )
    
    for layer in vgg19_model.layers[:-4]:  # Freeze all layers except the last 4
        layer.trainable = False
    
    x = vgg19_model.output
    # x = MaxPooling2D()(x)
    x = Flatten()(x)
    x = Dropout(0.5)(x)  # Dropout per regolarizzazione
    # x = Dense(64, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=vgg19_model.input, outputs=predictions)

    # Check the model summary
    model.summary()

    return model

def GenerateVGG16Model(num_classes: int) -> Model:
    # Load the VGG16 model without the top layer (i.e., the fully connected layers)
    vgg16_model = tf.keras.applications.VGG16(
        include_top=False,        # Exclude the top layer
        weights='imagenet',      # Load pre-trained weights from ImageNet
        input_shape=(256, 256, 3) # Specify input shape
    )
    
    # for layer in vgg16_model.layers[:-4]:  # Freeze all layers except the last 4
    #     layer.trainable = False
    vgg16_model.trainable = False

    x = vgg16_model.output
    x = GlobalAveragePooling2D()(x)
    # x = Flatten()(x)
    x = Dropout(0.5)(x)  # Dropout per regolarizzazione
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.25)(x)  # Dropout per regolarizzazione
    predictions = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=vgg16_model.input, outputs=predictions)

    # Check the model summary
    model.summary()

    return model

def GenerateResModel(num_classes: int) -> Model:
    # Load the VGG19 model without the top layer (i.e., the fully connected layers)
    res_model = resnet50.ResNet50(
        include_top=False,        # Exclude the top layer
        weights='imagenet',      # Load pre-trained weights from ImageNet
        input_shape=(256, 256, 3) # Specify input shape
    )
    
    ## freeze some layer 
    # res_model.trainable = True #all'inizio è già così
    # for layer in res_model.layers[:]:  # Freeze all layers 
    #     layer.trainable = False

    res_model.trainable = False

    x = res_model.output
    # x = MaxPooling2D()(x)
    # x = Flatten()(x)
    x = GlobalAveragePooling2D()(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.5)(x)  # Dropout per regolarizzazione
    predictions = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=res_model.input, outputs=predictions)

    # Check the model summary
    model.summary()

    return model

def GenerateCNN(num_classes: int) -> Model:
    # Definizione dell'input
    input_layer = tf.keras.Input(shape=(256, 256, 3))

    # Primo livello convoluzionale
    x = layers.Conv2D(filters=32,               # Numero di filtri
                    kernel_size=(3, 3),         # Dimensione del kernel
                    strides=(1, 1),             # Passo di convoluzione
                    padding='same',             # Padding (stessa dimensione dell'input)
                    activation='relu')(input_layer)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)  # Dropout 
    # Secondo livello convoluzionale
    x = layers.Conv2D(filters=64,               # Numero di filtri
                    kernel_size=(3, 3),         # Dimensione del kernel
                    strides=(1, 1),             # Passo di convoluzione
                    padding='same',             # Padding (stessa dimensione dell'input)
                    activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)  # Dropout 
    # Twerzo livello convoluzionale
    x = layers.Conv2D(filters=128,               # Numero di filtri
                    kernel_size=(3, 3),         # Dimensione del kernel
                    strides=(1, 1),             # Passo di convoluzione
                    padding='same',             # Padding (stessa dimensione dell'input)
                    activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)  # Dropout 
    # Flatten per passare al classificatore denso
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)  # Dropout 
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.5)(x)  # Dropout 
    predictions = Dense(num_classes, activation='softmax')(x)  # Output con 10 classi

    model = Model(inputs=input_layer, outputs=predictions)
    # Check the model summary
    model.summary()

    return model