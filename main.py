# -*- coding: utf-8 -*-
"""
CNN para Classificação de Cães vs Gatos - Trabalho Completo
"""
# %% [markdown]
# ## 1. Configuração Inicial
# ### Importações necessárias

# %%
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# %% [markdown]
# ## 2. Parâmetros Globais

# %%
IMG_SIZE = (150, 150)
BATCH_SIZE = 32
EPOCHS = 25
DATA_PATH = "dataset"

# %% [markdown]
# ## 3. Preparação dos Dados
# ### 3.1 Carregamento e divisão dos dados

# %%
print("=== PREPARAÇÃO DOS DADOS ===")

# Carregar metadados
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

# %% [markdown]
# ## 4. Pré-processamento
# ### 4.1 Data Augmentation

# %%
print("\n=== PRÉ-PROCESSAMENTO ===")

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

val_test_datagen = ImageDataGenerator(rescale=1. / 255)

# %% [markdown]
# ### 4.2 Geradores de dados

# %%
# Geradores
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

# %% [markdown]
# ## 5. Construção da CNN
# ### 5.1 Arquitetura da rede

# %%
print("\n=== CONSTRUÇÃO DA CNN ===")


def build_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
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

# %% [markdown]
# ## 6. Treinamento
# ### 6.1 Callbacks e execução

# %%
print("\n=== TREINAMENTO ===")

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

checkpoint = ModelCheckpoint(
    'best_model.h5',
    monitor='val_accuracy',
    save_best_only=True
)

history = model.fit(
    train_generator,
    steps_per_epoch=len(train_df) // BATCH_SIZE,
    validation_data=val_generator,
    validation_steps=len(val_df) // BATCH_SIZE,
    epochs=EPOCHS,
    callbacks=[early_stop, checkpoint]
)

# %% [markdown]
# ## 7. Avaliação
# ### 7.1 Métricas e gráficos

# %%
print("\n=== AVALIAÇÃO ===")

# Gráficos
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

# Avaliação no teste
test_loss, test_acc = model.evaluate(test_generator)
y_pred = (model.predict(test_generator) > 0.5).astype("int32")

print("\nRelatório de Classificação no Teste:")
print(classification_report(test_generator.classes, y_pred, target_names=['cats', 'dogs
