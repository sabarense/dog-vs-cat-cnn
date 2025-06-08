# -*- coding: utf-8 -*-

import os
import math
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, precision_score, balanced_accuracy_score

# Configurações
IMG_SIZE = (150, 150)
BATCH_SIZE = 32
EPOCHS = 25
DATA_PATH = "dataset"

def main():
    # 1. Verificar/criar estrutura de pastas
    for folder in ['train', 'test1', 'external_test']:
        path = os.path.join(DATA_PATH, folder)
        if not os.path.exists(path):
            os.makedirs(path)
            print(f"Pasta criada: {path}")

    # 2. Preparação dos dados
    print("=== PREPARAÇÃO DOS DADOS ===")
    train_files = [f for f in os.listdir(os.path.join(DATA_PATH, "train")) if f.endswith('.jpg')]
    train_labels = ['cat' if f.startswith('cat') else 'dog' for f in train_files]

    full_df = pd.DataFrame({
        'filename': train_files,
        'class': train_labels
    })

    # Divisão 70-15-15
    train_df, temp_df = train_test_split(
        full_df,
        test_size=0.3,
        stratify=full_df['class'],
        random_state=42
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        stratify=temp_df['class'],
        random_state=42
    )
    print(f"Treino: {len(train_df)} | Validação: {len(val_df)} | Teste: {len(test_df)}")

    # 3. Pré-processamento
    print("\n=== PRÉ-PROCESSAMENTO ===")
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )
    val_test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_dataframe(
        train_df,
        directory=os.path.join(DATA_PATH, "train"),
        x_col='filename',
        y_col='class',
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary'
    )
    val_generator = val_test_datagen.flow_from_dataframe(
        val_df,
        directory=os.path.join(DATA_PATH, "train"),
        x_col='filename',
        y_col='class',
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary'
    )
    test_generator = val_test_datagen.flow_from_dataframe(
        test_df,
        directory=os.path.join(DATA_PATH, "train"),
        x_col='filename',
        y_col='class',
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        shuffle=False
    )

    # 4. Construção da CNN
    print("\n=== CONSTRUÇÃO DA CNN ===")
    def build_model():
        model = Sequential([
            Input(shape=(150, 150, 3)),
            Conv2D(32, (3, 3), activation='relu'),
            MaxPooling2D(2, 2),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D(2, 2),
            Conv2D(128, (3, 3), activation='relu'),
            MaxPooling2D(2, 2),
            Flatten(),
            Dense(512, activation='relu'),
            Dropout(0.5),
            Dense(1, activation='sigmoid')
        ])
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        return model

    model = build_model()
    model.summary()

    # 5. Treinamento
    print("\n=== TREINAMENTO ===")
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )
    checkpoint = ModelCheckpoint(
        'best_model.keras',
        monitor='val_accuracy',
        save_best_only=True
    )
    steps_per_epoch = math.ceil(len(train_df) / BATCH_SIZE)
    validation_steps = math.ceil(len(val_df) / BATCH_SIZE)
    history = model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_generator,
        validation_steps=validation_steps,
        epochs=EPOCHS,
        callbacks=[early_stop, checkpoint]
    )

    # 6. Avaliação
    print("\n=== AVALIAÇÃO ===")
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Treino')
    plt.plot(history.history['val_accuracy'], label='Validação')
    plt.title('Acurácia por Época')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Treino')
    plt.plot(history.history['val_loss'], label='Validação')
    plt.title('Loss por Época')
    plt.legend()
    plt.savefig('training_metrics.png')
    plt.show()

    test_loss, test_acc = model.evaluate(test_generator)
    y_pred = (model.predict(test_generator) > 0.5).astype(int).flatten()
    balanced_acc = balanced_accuracy_score(test_generator.classes, y_pred)
    print(f"\nAcurácia no Teste: {test_acc:.2%}")
    print(f"Precisão Global: {precision_score(test_generator.classes, y_pred):.2%}")
    print(f"Acurácia Balanceada: {balanced_acc:.2%}")
    print("\nRelatório Detalhado:")
    print(classification_report(test_generator.classes, y_pred, target_names=['cats', 'dogs']))

    # 7. Salvamento do modelo
    model.save('dogs_vs_cats_final.keras')
    print("\nModelo salvo como 'dogs_vs_cats_final.keras'")

if __name__ == "__main__":
    main()
