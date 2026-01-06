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


# ------------------------------- Classification Binaire Chat ou Chien ------------------------------- #


# ------------------------------- Classification Fine ------------------------------- #

# Réseaux :

# Multi-layer perceptron simple
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

# Entrainement MLP
def Entrainement_nn(nn,epochs,train_generator, validation_generator, lr=1e-4):
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

# CNN simple
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
def VGG16_model(nom, img_height, img_width, trainable="block5"):
    conv_base = VGG16(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))
    if trainable is not None:
        for layer in conv_base.layers:      # Si on fait du fine-tuning, on dégèle uniquement les couches du block5 qui correspondent aux plus haut niveau des features (objets sémantiques, etc..)
            layer.trainable = layer.name.startswith(trainable)

    model = Sequential(name=nom)
    model.add(Input(shape=(img_height, img_width, 3)))
    model.add(conv_base)
    model.add(GlobalAveragePooling2D())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(37, activation='softmax'))
    return model


# ResNet50
def ResNet50_model(nom, img_height, img_width, fine_tune_stage="conv5"):
    conv_base = ResNet50(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))
    if fine_tune_stage is not None:
        for layer in conv_base.layers:      # Si on fait du fine-tuning, on dégèle uniquement les couches qui correspondent aux plus haut niveau des features (objets sémantiques, etc..)
            layer.trainable = layer.name.startswith(fine_tune_stage)

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



#### Affichage ####

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
def Analyse_resultats(nn,nn_history, train_generator, validation_generator):
    t_prediction_nn = time.time()
    score_nn_train = nn.evaluate(train_generator, verbose=1)
    score_nn_validation = nn.evaluate(validation_generator, verbose=1)
    predict_nn = nn.predict(validation_generator)

    y_true = validation_generator.classes
    y_pred = np.argmax(predict_nn, axis=1)

    t_prediction_nn = time.time() - t_prediction_nn

    print('Train accuracy:', score_nn_train[1])
    print('Validation accuracy:', score_nn_validation[1])
    print("Time Prediction: %.2f seconds" % t_prediction_nn)

    cm_norm = confusion_matrix(y_true, y_pred, normalize='false')
    plt.figure(figsize=(10,8))
    sns.heatmap(cm_norm, cmap="Blues")

    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()

    plot_training_analysis(nn_history)
    return t_prediction_nn

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

            # Load image
            img = load_img(img_path, target_size=self.img_size)
            img = img_to_array(img) / 255.0

            # Load mask (trimap)
            mask = load_img(mask_path,
                            target_size=self.img_size,
                            color_mode="grayscale")
            mask = img_to_array(mask).astype(np.int32)

            # Convert 1,2,3 -> 0,1 (binarisation: animal vs fond)
            mask = np.where(mask == 1, 1, 0)

            # Data augmentation synchronisée
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

# L'encodeur est pré-entrainé sur ImageNet. Pour l'instant on fixe les poids
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
    skips = reversed(skips[:-1])  # on enlève le bottleneck
    
    # Decoder avec skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        concat = Concatenate()
        x = concat([x, skip])
    
    # Dernière couche : upsampling + sigmoid pour segmentation binaire
    x = UpSampling2D((2, 2))(x)  # 64x64 -> 128x128
    last = Conv2D(output_channels, 3, activation='sigmoid', padding='same')(x)
    
    return tf.keras.Model(inputs=inputs, outputs=last)

# Affichage analyse des résultats pour la segmentation (Courbes apprentissage, métriques pertinentes pour segmentation)
def Analyse_resultats_segmentation(nn,nn_history, train_generator, validation_generator):
    # Calculer des métriques plus pertinentes pour la segmentation
    all_y_true = []
    all_y_pred = []

    # Prédire sur tous les batches de validation
    for i in range(len(validation_generator)):
        X_batch, y_batch = validation_generator[i]
        y_pred_batch = nn.predict(X_batch, verbose=0)

        # Binariser à 0.5
        y_pred_binary = (y_pred_batch > 0.5).astype(np.int32)
        
        # Aplatir et accumuler
        all_y_true.extend(y_batch.flatten())
        all_y_pred.extend(y_pred_binary.flatten())

    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)

    iou = jaccard_score(all_y_true, all_y_pred)
    precision = precision_score(all_y_true, all_y_pred, zero_division=0)
    recall = recall_score(all_y_true, all_y_pred, zero_division=0)
    f1 = f1_score(all_y_true, all_y_pred, zero_division=0)
    dice = 2 * (np.sum(all_y_pred * all_y_true)) / (np.sum(all_y_pred) + np.sum(all_y_true) + 1e-8)

    print("="*50)
    print("MÉTRIQUES DE SEGMENTATION")
    print("="*50)
    print(f"Accuracy pixel-wise:  {np.mean(all_y_true == all_y_pred):.4f} (défaut Keras)")
    print(f"IoU (Jaccard):        {iou:.4f}")
    print(f"Dice coefficient:     {dice:.4f}")
    print(f"Precision (animal):   {precision:.4f}")
    print(f"Recall (animal):      {recall:.4f}")
    print(f"F1-score:             {f1:.4f}")
    print("="*50)
    print("\nInterprétation:")
    print(f"- IoU/Dice proches de 1 = bon overlap entre prédiction et réalité")
    print(f"- Precision = {precision:.4f} : {precision*100:.1f}% des pixels prédits comme animaux sont vrais")
    print(f"- Recall = {recall:.4f} : {recall*100:.1f}% des vrais pixels animaux sont détectés")

    t_prediction_nn = time.time()
    score_nn_train = nn.evaluate(train_generator, verbose=1)
    score_nn_validation = nn.evaluate(validation_generator, verbose=1)

    t_prediction_nn = time.time() - t_prediction_nn

    print('Train accuracy:', score_nn_train[1])
    print('Validation accuracy:', score_nn_validation[1])
    print("Time Prediction: %.2f seconds" % t_prediction_nn)

    plt.show()

    plot_training_analysis(nn_history)
    return t_prediction_nn
