# CNN for classification on Grapevine leaves
from importLibrary import *

INPUT_SHAPE=(256, 256, 3)

def OutputNetwork(num_classes: int, input_layer: layers) -> layers:
    """
    Crea una rete neurale che può essere utilizzata come testa finale per un modello,
    aggiungendo strati densi e uno strato di classificazione softmax.

    :param num_classes: Numero di classi per la classificazione, corrispondente ai nodi di output finali.
    :param x: Input layer o output di un modello pre-addestrato su cui aggiungere questa testa finale.
    :return: Un modello Keras che rappresenta la testa finale, pronto per essere unito ad altri modelli.
    """

    # Layer possibili:
    # x = GlobalAveragePooling2D()(x)

    # Flatten: appiattisce l'input (es. una mappa di attivazione 2D) in un vettore 1D.
    x = Flatten()(input_layer)
    # Primo strato Dropout per la regolarizzazione, con tasso di abbandono del 50%.
    x = Dropout(0.5)(x)
    
    # Strato Dense di 256 neuroni
    x = Dense(256)(x)
    # BatchNormalization per normalizzare l'output del livello Dense, migliorando la stabilità dell'allenamento.
    x = BatchNormalization()(x)
    #Activation relu applicata dopo BatchNorm per normalizzare sul lineare e non perdere informazioni
    x = Activation('relu')(x)
    # Secondo strato Dropout per ridurre l'overfitting, sempre con tasso di abbandono del 50%.
    x = Dropout(0.5)(x)
    
    # Strato Dense di 120 neuroni con attivazione ReLU, che aggiunge ulteriore capacità di apprendimento.
    x = Dense(120)(x)
    # BatchNormalization per migliorare la velocità e stabilità dell'allenamento in questo livello.
    x = BatchNormalization()(x)
    #Activation relu applicata dopo BatchNorm per normalizzare sul lineare e non perdere informazioni
    x = Activation('relu')(x)
    # Terzo strato Dropout per regolarizzazione, con tasso di abbandono del 35%.
    x = Dropout(0.35)(x)
    
    # Strato Dense di 60 neuroni con attivazione ReLU, ultimo strato denso prima dell'output.
    x = Dense(60)(x)
    # BatchNormalization per normalizzare l'output dell'ultimo livello Dense.
    x = BatchNormalization()(x)
    #Activation relu applicata dopo BatchNorm per normalizzare sul lineare e non perdere informazioni
    x = Activation('relu')(x)

    # Strato finale Dense per la classificazione: num_classes neuroni con attivazione softmax,
    # per ottenere una distribuzione di probabilità sulle classi.
    predictions = Dense(num_classes, activation='softmax')(x)
    
    # Creazione del modello, che prende come input 'x' e restituisce 'predictions'.
    #model = Model(inputs=x, outputs=predictions, name="OutputNetwork")
    
    return predictions

def GenerateVGG19Model(num_classes: int) -> Model:
    # Load the VGG19 model without the top layer (i.e., the fully connected layers)
    vgg19_model = tf.keras.applications.VGG19(
        include_top=False,        # Exclude the top layer
        weights='imagenet',      # Load pre-trained weights from ImageNet
        input_shape=INPUT_SHAPE # Specify input shape
    )
    
    # for layer in vgg19_model.layers[:-4]:  # Freeze all layers except the last 4
    #     layer.trainable = False
    vgg19_model.trainable = False

    x = vgg19_model.output

    output_layer = OutputNetwork(num_classes=num_classes, input_layer=x)    

    model = Model(inputs=vgg19_model.input, outputs=output_layer)

    # Check the model summary
    model.summary()

    return model

def GenerateVGG16Model(num_classes: int) -> Model:
    # Load the VGG16 model without the top layer (i.e., the fully connected layers)
    vgg16_model = tf.keras.applications.VGG16(
        include_top=False,        # Exclude the top layer
        weights='imagenet',      # Load pre-trained weights from ImageNet
        input_shape=INPUT_SHAPE # Specify input shape
    )
    
    for layer in vgg16_model.layers[:-4]:  # Freeze all layers except the last 4
        layer.trainable = False
    # vgg16_model.trainable = False

    x = vgg16_model.output
    output_layer = OutputNetwork(num_classes=num_classes, input_layer=x)    

    model = Model(inputs=vgg16_model.input, outputs=output_layer)

    # Check the model summary
    model.summary()

    return model

def GenerateResModel(num_classes: int) -> Model:
    # Load the VGG19 model without the top layer (i.e., the fully connected layers)
    res_model = resnet50.ResNet50(
        include_top=False,        # Exclude the top layer
        weights='imagenet',      # Load pre-trained weights from ImageNet
        input_shape=INPUT_SHAPE # Specify input shape
    )
    
    ## freeze some layer 
    # res_model.trainable = True #all'inizio è già così
    # for layer in res_model.layers[:]:  # Freeze all layers 
    #     layer.trainable = False

    res_model.trainable = False

    x = res_model.output
    
    output_layer = OutputNetwork(num_classes=num_classes, input_layer=x)    

    model = Model(inputs=res_model.input, outputs=output_layer)

    # Check the model summary
    model.summary()

    return model

def GenerateCNN(num_classes: int) -> Model:
    # Definizione dell'input
    input_layer = tf.keras.Input(shape=INPUT_SHAPE)

    # Primo livello convoluzionale
    x = layers.Conv2D(filters=32,               # Numero di filtri
                    kernel_size=(3, 3),         # Dimensione del kernel
                    strides=(1, 1),             # Passo di convoluzione
                    padding='same',             # Padding (stessa dimensione dell'input)
                    activation='linear')(input_layer)
    x = BatchNormalization()(x)  # Batch Normalization subito dopo Conv2D
    x = Activation('relu')(x)
    
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.5)(x)  # Dropout 
    
    # Secondo livello convoluzionale
    x = layers.Conv2D(filters=64,               # Numero di filtri
                    kernel_size=(3, 3),         # Dimensione del kernel
                    strides=(1, 1),             # Passo di convoluzione
                    padding='same',             # Padding (stessa dimensione dell'input)
                    activation='linear')(x)
    x = BatchNormalization()(x)  # Batch Normalization subito dopo Conv2D
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.5)(x)  # Dropout 
    
    # Terzo livello convoluzionale
    x = layers.Conv2D(filters=128,               # Numero di filtri
                    kernel_size=(3, 3),         # Dimensione del kernel
                    strides=(1, 1),             # Passo di convoluzione
                    padding='same',             # Padding (stessa dimensione dell'input)
                    activation='linear')(x)
    x = BatchNormalization()(x)  # Batch Normalization subito dopo Conv2D
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
  
    # Flatten per passare al classificatore denso
    x = Flatten()(x)
    x = Dropout(0.5)(x)  # Dropout 

    x = Dense(256)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    x = Dropout(0.35)(x)  # Dropout 
    x = Dense(120)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    x = Dropout(0.35)(x)  # Dropout 
    predictions = Dense(num_classes, activation='softmax')(x)  # Output con 10 classi

    model = Model(inputs=input_layer, outputs=predictions)
    # Check the model summary
    model.summary()

    return model

if __name__ == '__main__':
    num_classes=5
    model = GenerateVGG19Model(num_classes)
