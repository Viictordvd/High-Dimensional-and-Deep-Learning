# ------------------------------- Imports ------------------------------- #

#Librairies de base
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
from PIL import Image
import random

# TensorFlow
import tensorflow as tf

import time

# Séparation Test/Train/Validation
from sklearn.model_selection import train_test_split

# Optimizers
from tensorflow.keras.optimizers import Adam, Adadelta

# Metrics
from sklearn.metrics import confusion_matrix, jaccard_score, precision_score, recall_score, f1_score, roc_curve, auc, precision_recall_curve, roc_auc_score

# TensorFlow layers
from tensorflow.keras.layers import Input, Conv2D, MaxPool2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization, UpSampling2D, Concatenate, GlobalAveragePooling2D
from tensorflow.keras.models import Sequential

# Utiles TensorFlow
from tensorflow.keras.utils import load_img, img_to_array, Sequence
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import load_img, img_to_array,ImageDataGenerator

# Modèles pré-entraînés
from tensorflow.keras.applications import VGG16, ResNet50, MobileNetV2
from tensorflow.keras.applications.resnet50 import preprocess_input
import matplotlib.image as mpimg

# ------------------------------- Classification Binaire Chat ou Chien ------------------------------- #


### Réseaux ###

# premier CNN simple pour la classification binaire. Les images sont redimensionnées en img_height x img_width pixels pour pouvoir être traitées par le réseau
# Le réseau est composé de 3 couches de convolution + maxpooling, suivies de 2 couches denses avec dropout (pour éviter le surapprentissage) avant la couche de sortie.
# La fonction d'activation de la couche de sortie est une sigmoïde pour produire une probabilité entre 0 et 1 (chat ou chien).
def cnn_simple(nom,img_width, img_height):
    cnn = Sequential(name=nom)
    cnn.add(Input(shape=(img_height, img_width,3)))
    cnn.add(Conv2D(32, (3,3), activation='relu'))
    cnn.add(MaxPooling2D(pool_size=(2,2)))
    cnn.add(Conv2D(64, (3,3), activation='relu'))
    cnn.add(MaxPooling2D(pool_size=(2,2)))
    cnn.add(Conv2D(96, (3,3), activation='relu'))
    cnn.add(MaxPooling2D(pool_size=(2,2)))
    cnn.add(Flatten())              # Convertir la dimension de la sortie des couches convolutionnelles pour la couche dense (vecteur)
    cnn.add(Dense(64, activation='relu'))                # MLP simple (couche dense). 
    cnn.add(Dropout(0.5))
    cnn.add(Dense(1, activation='sigmoid'))               # activation sigmoid pour classification binaire
    return cnn

# VGG16, on utilise ici un CNN pré-entraîné auquel on rajoute une tête de classification binaire (MLP). 
# Comme VGG-16 est déjà entraîné, on bloque 'entraînement de ses poids (sauf dans le cas du fine-tuning où on dégèle les couches tardives de VGG-16 pour les ré-entraîner légèrement)
def VGG16_model_binaire(nom, img_height, img_width, trainable=False):
    conv_base = VGG16(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))
    
    # Pour le Fine-tuning on gere si les poids sont gelés ou non
    if trainable == True:         
        conv_base.trainable = True # On rend la base convolutive entraînable
        for layer in conv_base.layers[:15]: # On bloque les 15 premières couches (caractéristiques générales)
            layer.trainable = False
    else:
        conv_base.trainable = False

    model = Sequential(name=nom)
    model.add(Input(shape=(img_height, img_width, 3)))
    model.add(conv_base)
    model.add(GlobalAveragePooling2D())             # Transforme la sortie de VGG-16 pour correspondre à l'entrée du MLP. 
    # Meilleur que flatten car permet de conserver une meilleure représentation de l'info dans un vecteur
    model.add(Dense(256, activation='relu'))                # MLP simple. 
    model.add(Dense(1, activation='sigmoid'))               # activation sigmoid pour classification binaire
    return model


### Entrainement pour classif binaire ###

# Focntion d'entraînement. La loss adaptée pour la classification binaire est la binary crossentropy. L'optimiseur utilisé est Adam. La métrique optimisée est l'accuracy du modèle
def Entrainement_nn_binaire(cnn,epochs,train_generator,validation_generator,lr=3e-4): 
    print("Entrainement de ",cnn.name)
    cnn.compile(
        loss = 'binary_crossentropy',               # Pour classif binaire
        optimizer = Adam(learning_rate=lr),
        metrics = ['accuracy'])

    t_learning_cnn = time.time()
    cnn_history = cnn.fit(
        train_generator,
        validation_data = validation_generator,
        epochs = epochs)
    
    t_learning_cnn = time.time() - t_learning_cnn
    print("Learning time for %d epochs : %d seconds" % (epochs, t_learning_cnn))
    return t_learning_cnn, cnn_history


#### Affichage de l'entrainement ####

def plot_training_analysis(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(acc))

    plt.subplot(1,2,1)
    plt.plot(epochs, acc, 'b', linestyle="--",label='Training accuracy')
    plt.plot(epochs, val_acc, 'g', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(epochs, loss, 'b', linestyle="--",label='Training loss')
    plt.plot(epochs, val_loss,'g', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

    plt.tight_layout()
    plt.show()

# Analyse des résultats : courbes d'apprentissage, permet de vérifier si on a un sur-apprentissage par ex
def Analyse_resultats_binaire(cnn,cnn_history, train_generator, validation_generator):
    t_prediction_cnn = time.time()
    score_cnn_train = cnn.evaluate(train_generator, verbose=1)
    score_cnn_validation = cnn.evaluate(validation_generator, verbose=1)

    t_prediction_cnn = time.time() - t_prediction_cnn

    print('Train accuracy:', score_cnn_train[1])
    print('Validation accuracy:', score_cnn_validation[1])
    print("Time Prediction: %.2f seconds" % t_prediction_cnn)

    plot_training_analysis(cnn_history)
    return t_prediction_cnn

def predict_animal(model, df, img_dir, index=None):
    if index is None:
        index = np.random.randint(0, len(df))
    
    img_name = df.iloc[index]['Image']
    true_label = df.iloc[index]['SPECIES_NAME']
    img_path = os.path.join(img_dir, img_name)

    # 2. Charger et préparer l'image (doit être identique au target_size du generator)
    # On utilise target_size=(150, 150) comme défini dans votre code
    img = load_img(img_path, target_size=(150, 150))
    img_array = img_to_array(img) / 255.0  # Normalisation 1./255
    img_batch = np.expand_dims(img_array, axis=0)  # Ajouter la dimension de batch (1, 150, 150, 3)

    prediction = model.predict(img_batch)
    
    # Comme class_mode='binary', 0 est  'Cat' et 1 est 'Dog' 
    if prediction[0][0] > 0.5:
        res = "Dog"
        prob = prediction[0][0]
    else:
        res = "Cat"
        prob = 1 - prediction[0][0]

    plt.figure(figsize=(6, 6))
    plt.imshow(img)
    color = "green" if res == true_label else "red"
    plt.title(f"Réel: {true_label}\nPred: {res} ({prob:.2%})", color=color, fontsize=14, fontweight="bold")
    plt.axis("off")
    plt.show()

# ------------------------------- Classification Fine ------------------------------- #


### Réseaux ###

# Multi-layer perceptron simple
# Ce réseau sert de baseline pour la classif fine, on va réduire drastiquement la taille des images à 32x32 pixels pour limiter le nombre de paramètres du réseau à apprendre.
# Néanmoins, le nombre de paramètres reste élevé (plus de 300k) pour un MLP, on s'attend donc à un surapprentissage rapide.
def mlp(nom, N_classes=37):
    mlp = Sequential(name=nom)
    mlp.add(Input(shape=(32, 32, 3)))
    mlp.add(Flatten())
    mlp.add(Dense(128, activation='relu'))
    mlp.add(Dropout(0.4))
    mlp.add(Dense(128, activation='relu'))
    mlp.add(Dropout(0.4))
    mlp.add(Dense(N_classes, activation='softmax'))
    return mlp

# CNN simple
# trrès similaire au cas de la classification binaire à l'exception faite que la sortie du modèle est un vecteur de la taille du nombre de classes (37),
#  de telle sorte que la i-ème coordonée du vecteur corresponde à la probabilité de l'image d'appartenir à la i-ème classe.
# La fonction d'activation de la sortie est un softmax pour s'assurer qu'on ait des probabilités en sortie (somme = 1).
def CNN(nom, img_width, img_height, N_classes=37):
    model = Sequential(name=nom)
    model.add(Input(shape=(img_height, img_width, 3)))

    model.add(Conv2D(32, 3, padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D())
    model.add(Conv2D(64, 3, padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D())
    model.add(Conv2D(128, 3, padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D())

    model.add(GlobalAveragePooling2D())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(N_classes, activation='softmax'))
    return model

# VGG16
# De même, modèle similaire au cas binaire, à l'exception de la sortie.
def VGG16_model_classif_fine(nom, img_height, img_width, trainable="block5"):
    conv_base = VGG16(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))

    # Pour le Fine-tuning on gere si les poids sont gelés ou non
    if trainable == True:         
        conv_base.trainable = True # On rend la base convolutive entraînable
        for layer in conv_base.layers[:15]: # On bloque les 15 premières couches (caractéristiques générales)
            layer.trainable = False
    else:
        conv_base.trainable = False
    
    model = Sequential(name=nom)
    model.add(Input(shape=(img_height, img_width, 3)))
    model.add(conv_base)
    model.add(GlobalAveragePooling2D())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(37, activation='softmax'))
    return model


# ResNet50
# On utilise ici un autre modèle pré-entraîné : ResNet50. Ce modèle fait partie de la famille des modèles ResNet, il comporte 50 couches. 
# Il utilise des connections résiduelles.
def ResNet50_model(nom, img_height, img_width, fine_tune_stage="conv5"):
    conv_base = ResNet50(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))
    if fine_tune_stage is not None:
        for layer in conv_base.layers:      # Si on fait du fine-tuning, on dégèle uniquement les couches qui correspondent aux plus haut niveau des features (objets sémantiques, etc..)
            layer.trainable = layer.name.startswith(fine_tune_stage)
    else:
        conv_base.trainable=False

    model = Sequential(name=nom)
    model.add(Input(shape=(img_height, img_width, 3)))
    model.add(conv_base)
    model.add(GlobalAveragePooling2D())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(37, activation='softmax'))
    return model

# MobileNetV2
def MobileNetV2_model(nom, img_height, img_width, fine_tune_from="block_13"):
    conv_base = MobileNetV2(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3),name="mobilenet_backbone")
    if fine_tune_from is not None:
        for layer in conv_base.layers:
            if layer.name.startswith(fine_tune_from) or layer.name.startswith("Conv_1"):
                layer.trainable = True
            else:
                layer.trainable = False

    model = Sequential(name=nom)
    model.add(Input(shape=(img_height, img_width, 3)))
    model.add(conv_base)
    model.add(GlobalAveragePooling2D())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(37, activation='softmax'))
    return model


### Entrainement pour classif fine ###

def Entrainement_nn_fine(nn,epochs,train_generator, validation_generator, lr=1e-4):
    print("Entrainement de ",nn.name)
    nn.compile(
        loss = 'sparse_categorical_crossentropy',
        optimizer = Adam(learning_rate=lr),
        metrics = ['accuracy'])
    
    t_learning_nn = time.time()
    nn_history = nn.fit(train_generator, 
                          validation_data = validation_generator, 
                          epochs = epochs)

    t_learning_nn = time.time() - t_learning_nn
    print("Learning time for %d epochs : %d seconds" % (epochs, t_learning_nn))
    return t_learning_nn, nn_history


#### Affichage de l'entrainement ####

# Courbes apprentissage
def plot_training_analysis(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(acc))

    plt.subplot(1,2,1)
    plt.plot(epochs, acc, 'b', linestyle="--",label='Training accuracy')
    plt.plot(epochs, val_acc, 'g', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(epochs, loss, 'b', linestyle="--",label='Training loss')
    plt.plot(epochs, val_loss,'g', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

    plt.tight_layout()
    plt.show()

# Affichage analyse des résultats (Matrice de confusion + courbes apprentissage)
def Analyse_resultats(nn,nn_history, train_generator, validation_generator,confusion_races=False):
    validation_generator.shuffle = False
    validation_generator.reset()
    t_prediction_nn = time.time()
    score_nn_train = nn.evaluate(train_generator, verbose=1)
    score_nn_validation = nn.evaluate(validation_generator, verbose=1)
    predict_nn = nn.predict(validation_generator)

    y_true = validation_generator.classes
    y_pred = np.argmax(predict_nn, axis=1)

    class_labels = {v: k for k, v in validation_generator.class_indices.items()}

    t_prediction_nn = time.time() - t_prediction_nn

    print('Train accuracy:', score_nn_train[1])
    print('Validation accuracy:', score_nn_validation[1])
    print("Time Prediction: %.2f seconds" % t_prediction_nn)

    cm = confusion_matrix(y_true, y_pred)

    confused_pairs = []
    num_classes = len(class_labels)
    
    for i in range(num_classes):
        for j in range(num_classes):
            if i != j:
                count = cm[i, j]
                if count > 0:
                    true_breed = class_labels[i]
                    pred_breed = class_labels[j]
                    confused_pairs.append({
                        'Vraie Race': true_breed,
                        'Race Prédite': pred_breed,
                        'Nombre d\'erreurs': count
                    })

    df_confusion = pd.DataFrame(confused_pairs)
    df_confusion = df_confusion.sort_values(by='Nombre d\'erreurs', ascending=False)

    plt.figure(figsize=(10,8))
    sns.heatmap(
        cm,
        annot=True,      # Affiche les nombres
        fmt="d",         # Format décimal (entier)
        cmap="Blues", 
        annot_kws={"size": 8} # Police plus petite pour que ça rentre dans les cases
    )

    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()

    plot_training_analysis(nn_history)
    validation_generator.shuffle = True
    if confusion_races == True:
        print(f"\nTotal des paires de races confondues : {len(df_confusion)}")
        print("\n 15 confusions les plus fréquentes")
        print(df_confusion.head(15).to_string(index=False))

    return t_prediction_nn

def predict_breed_samples(model, df, img_dir, train_generator, img_height, img_width, n_samples=5):
    samples = df.sample(n_samples)
    
    # 2. Récupérer les noms des classes (races) 
    # Important : Keras trie les classes par ordre alphabétique par défaut
    class_indices = train_generator.class_indices
    labels = {v: k for k, v in class_indices.items()} # Inverse le dico : {0: 'Abyssinian', ...}

    plt.figure(figsize=(20, 4))
    
    for i, (idx, row) in enumerate(samples.iterrows()):
        img_path = os.path.join(img_dir, row['Image'])
        true_breed = row['BREED_NAME']
        
        img = load_img(img_path, target_size=(img_height, img_width))
        img_array = img_to_array(img) / 255.0
        img_batch = np.expand_dims(img_array, axis=0)

        # Prédiction
        preds = model.predict(img_batch, verbose=0)
        pred_idx = np.argmax(preds[0]) # On prend l'indice de la probabilité max
        pred_breed = labels[pred_idx]
        confidence = preds[0][pred_idx]

        plt.subplot(1, n_samples, i + 1)
        plt.imshow(img)
        color = "green" if pred_breed.lower() == true_breed.lower() else "red"
        plt.title(f"Réel: {true_breed}\nPred: {pred_breed}\n({confidence:.1%})", 
                  color=color, fontsize=10)
        plt.axis("off")
    
    plt.tight_layout()
    plt.show()


def plot_confused_pairs(pairs, df, base_dir, path_col='Image', label_col='BREED_NAME'):
    n_pairs = len(pairs)
    plt.figure(figsize=(10, 4 * n_pairs))
    
    for i, (true_race, confused_race) in enumerate(pairs):
        filename_true = df[df[label_col] == true_race][path_col].sample(1).values[0]
        filename_confused = df[df[label_col] == confused_race][path_col].sample(1).values[0]
        
        true_img_path = os.path.join(base_dir, filename_true)
        confused_img_path = os.path.join(base_dir, filename_confused)

        ax1 = plt.subplot(n_pairs, 2, 2*i + 1)
        try:
            img1 = mpimg.imread(true_img_path)
            ax1.imshow(img1)
            ax1.set_title(f"Vraie classe :\n{true_race}", color='green')
        except FileNotFoundError:
            ax1.text(0.5, 0.5, "Image introuvable", ha='center')
            print(f"Erreur: Impossible de trouver {true_img_path}")
        ax1.axis('off')
        
        ax2 = plt.subplot(n_pairs, 2, 2*i + 2)
        try:
            img2 = mpimg.imread(confused_img_path)
            ax2.imshow(img2)
            ax2.set_title(f"Classe confondue avec :\n{confused_race}", color='red')
        except FileNotFoundError:
            ax2.text(0.5, 0.5, "Image introuvable", ha='center')
        ax2.axis('off')

    plt.tight_layout()
    plt.show()

# ------------------------------- Segmentation ------------------------------- #

# Définition classe dataLoader pour la segmentation
class SegmentationDataLoader(Sequence):
    def __init__(self, df, img_dir, mask_dir,
                 img_size=(128,128), batch_size=20,
                 augment=False):

        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_size = img_size
        self.batch_size = batch_size
        self.augment = augment

    def __len__(self):
        return len(self.df) // self.batch_size

    def __getitem__(self, idx):
        batch_df = self.df.iloc[idx*self.batch_size : (idx+1)*self.batch_size]

        X = []
        Y = []

        for _, row in batch_df.iterrows():
            img_path = os.path.join(self.img_dir, row["Image"])
            mask_path = os.path.join(self.mask_dir,
                                     row["Image"].replace(".jpg", ".png"))

            img = load_img(img_path, target_size=self.img_size)
            img = img_to_array(img) / 255.0

            mask = load_img(mask_path,
                            target_size=self.img_size,
                            color_mode="grayscale")
            mask = img_to_array(mask).astype(np.int32)

            mask = np.where(mask == 1, 1, 0)

            if self.augment:
                if random.random() < 0.5:
                    img = np.fliplr(img)
                    mask = np.fliplr(mask)

            X.append(img)
            Y.append(mask)

        return np.array(X), np.array(Y)
    

# Entrainement réseau de segmentation
def Entrainement_nn_segmentation(nn,epochs,train_generator, validation_generator, lr=1e-4):
    print("Entrainement de ",nn.name)
    nn.compile(
        loss = 'binary_crossentropy',
        optimizer = Adam(learning_rate=lr),
        metrics = ['accuracy'])
    
    t_learning_nn = time.time()
    nn_history = nn.fit(train_generator, 
                          validation_data = validation_generator, 
                          epochs = epochs,
                          verbose=1)

    t_learning_nn = time.time() - t_learning_nn
    print("Learning time for %d epochs : %d seconds" % (epochs, t_learning_nn))
    return t_learning_nn, nn_history

# Réseau U-Net avec MobileNetV2 comme encodeur:

# On extrait les couches de MobileNetV2 pour la partie encodeur du U-Net
base_model = tf.keras.applications.MobileNetV2(input_shape=[128, 128, 3], include_top=False)
layer_names = [
    'block_1_expand_relu',   # 64x64
    'block_3_expand_relu',   # 32x32
    'block_6_expand_relu',   # 16x16
    'block_13_expand_relu',  # 8x8
    'block_16_project',      # 4x4 (Bottleneck)
]
base_model_outputs = [base_model.get_layer(name).output for name in layer_names]

# L'encodeur est pré-entrainé sur ImageNet.
down_stack = tf.keras.Model(inputs=base_model.input, outputs=base_model_outputs)
down_stack.trainable = False

# Le décodeur contient des couches de upsampling + Conv2D
# Fonction pour créer les blocs de décoder
def upsample_block(filters):
    return tf.keras.Sequential([
        UpSampling2D((2, 2)),
        Conv2D(filters, 3, activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(filters, 3, activation='relu', padding='same'),
        BatchNormalization()
    ])

up_stack = [
    upsample_block(512),  # 4x4 -> 8x8
    upsample_block(256),  # 8x8 -> 16x16
    upsample_block(128),  # 16x16 -> 32x32
    upsample_block(64),   # 32x32 -> 64x64
]

def unet_model(output_channels: int = 1):
    inputs = tf.keras.layers.Input(shape=[128, 128, 3])
    
    # Encoder
    skips = down_stack(inputs)
    x = skips[-1]
    skips = reversed(skips[:-1])
    
    # Decoder avec skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        concat = Concatenate()
        x = concat([x, skip])
    
    # Dernière couche : upsampling + sigmoid pour segmentation binaire
    x = UpSampling2D((2, 2))(x)  # 64x64 -> 128x128
    last = Conv2D(output_channels, 3, activation='sigmoid', padding='same')(x)
    
    return tf.keras.Model(inputs=inputs, outputs=last)

def Segmentation_Scores(val_loader_seg,Unet):
    all_y_true = []
    all_y_pred = []

    for i in range(len(val_loader_seg)):
        X_batch, y_batch = val_loader_seg[i]
        y_pred_batch = Unet.predict(X_batch, verbose=0)
        y_pred_binary = (y_pred_batch > 0.5).astype(np.int32)

        all_y_true.extend(y_batch.flatten())
        all_y_pred.extend(y_pred_binary.flatten())

    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)

    iou = jaccard_score(all_y_true, all_y_pred)
    precision = precision_score(all_y_true, all_y_pred, zero_division=0)
    recall = recall_score(all_y_true, all_y_pred, zero_division=0)

    dice = 2 * (np.sum(all_y_pred * all_y_true)) / (np.sum(all_y_pred) + np.sum(all_y_true) + 1e-8)

    print("MÉTRIQUES DE SEGMENTATION \n")
    print(f"Accuracy pixel-wise:  {np.mean(all_y_true == all_y_pred):.4f} (défaut)")
    print(f"IoU (Jaccard):        {iou:.4f}")
    print(f"Dice coefficient:     {dice:.4f}")
    print(f"Precision:   {precision:.4f}")
    print(f"Recall:      {recall:.4f}")

def Visualisation_Prediction_U_Net(model,val_loader,num_examples):
    fig, axes = plt.subplots(num_examples, 3, figsize=(15, 5*num_examples))

    # Prendre num_examples du validation loader
    for i in range(num_examples):
        X_batch, y_batch = val_loader[i]
        
        # Prédire les masques
        y_pred = model.predict(X_batch[:1])  # Prendre la première image du batch
        
        # Récupérer les valeurs
        img = X_batch[0]
        mask_true = y_batch[0]
        mask_pred = y_pred[0]
        
        # Afficher l'image
        axes[i, 0].imshow(img)
        axes[i, 0].set_title(f'Image {i+1}', fontweight='bold')
        axes[i, 0].axis('off')
        
        # Afficher le masque réel
        axes[i, 1].imshow(mask_true[:,:,0], cmap='gray')
        axes[i, 1].set_title(f'Masque réel {i+1}', fontweight='bold')
        axes[i, 1].axis('off')
        
        # Afficher le masque prédit
        axes[i, 2].imshow(mask_pred[:,:,0], cmap='gray')
        axes[i, 2].set_title(f'Masque prédit {i+1}', fontweight='bold')
        axes[i, 2].axis('off')

    plt.tight_layout()
    plt.show()

def comparaison_races (Unet,img_dir,mask_dir,validation_df_seg,df):
    iou_scores = []

    THRESHOLD = 0.5 
    for idx, row in validation_df_seg.iterrows():
        img_path = os.path.join(img_dir, row["Image"])
        img = load_img(img_path, target_size=(128, 128))
        x = img_to_array(img) / 255.0
        x = np.expand_dims(x, axis=0)
        mask_path = os.path.join(mask_dir, row["Image"].replace(".jpg", ".png"))
        mask = load_img(mask_path, target_size=(128, 128), color_mode="grayscale")
        mask = img_to_array(mask).astype(np.int32)
        y_true = np.where(mask == 1, 1, 0).astype(np.float32)

        y_pred = Unet.predict(x, verbose=0)[0] # Résultat (128, 128, 1)

        y_pred_bin = (y_pred > THRESHOLD).astype(np.float32)
        
        intersection = np.sum(y_true * y_pred_bin)
        union = np.sum(y_true) + np.sum(y_pred_bin) - intersection
        
        iou = 1.0 if union == 0 else intersection / union
        iou_scores.append(iou)

    df_results = validation_df_seg.copy()
    df_infos = df[['Image', 'BREED_NAME', 'SPECIES_NAME']].drop_duplicates(subset=['Image'])
    df_results = df_results.merge(df_infos, on='Image', how='left')
    df_results['IoU'] = iou_scores

    print("\n CHATS vs CHIENS (IoU Moyen)")
    species_stats = df_results.groupby('SPECIES_NAME')['IoU'].mean().reset_index()
    print(species_stats)
    print("\n Classement races par qualité de segmentation (IoU)")
    breed_stats = df_results.groupby('BREED_NAME')['IoU'].mean().reset_index()
    breed_stats = breed_stats.sort_values(by='IoU', ascending=False)
    print("\n 10 races les mieux segmentées")
    print(breed_stats.head(10).to_string(index=False))
    print("\n 10 races les moins bien segmentées")
    print(breed_stats.tail(10).to_string(index=False))
    return df_results

def visualize_specific_breeds(race_list, df, model, img_dir, mask_dir, group_name="Groupe"):
    print(f"\nVisualisation : {group_name}")
    
    for breed in race_list:
        breed_df = df[df['BREED_NAME'] == breed]
        
        if len(breed_df) == 0:
            print(f"Pas d'images trouvées pour {breed}")
            continue

        row = breed_df.sample(1).iloc[0]

        img_path = os.path.join(img_dir, row['Image'])
        mask_filename = row['Image'].replace(".jpg", ".png")
        mask_path = os.path.join(mask_dir, mask_filename)
        
        try:
            img_origin = load_img(img_path, target_size=(128, 128))
            x = img_to_array(img_origin) / 255.0
            x_input = np.expand_dims(x, axis=0)
            mask_true = load_img(mask_path, target_size=(128, 128), color_mode="grayscale")
            mask_true = img_to_array(mask_true).astype(np.int32)
            mask_true_bin = np.where(mask_true == 1, 1, 0)
            pred_prob = model.predict(x_input, verbose=0)[0]
            pred_mask = (pred_prob > 0.5).astype(np.float32)
            plt.figure(figsize=(14, 3))

            plt.subplot(1, 3, 1)
            plt.imshow(img_origin)
            plt.title(f"{breed}\n(Image Originale)")
            plt.axis('off')
            
            plt.subplot(1, 3, 2)
            plt.imshow(mask_true_bin, cmap='gray')
            plt.title("Vérité Terrain (Masque)")
            plt.axis('off')

            plt.subplot(1, 3, 3)
            plt.imshow(pred_mask, cmap='gray')
            plt.title(f"Prédiction")
            plt.axis('off')
            
            plt.show()
            
        except Exception as e:
            print(f"Erreur lors de l'affichage pour {breed}: {e}")