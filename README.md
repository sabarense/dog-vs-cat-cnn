# # Dogs vs Cats with CNN: Image Classification Project

Este projeto implementa um pipeline completo para classificação binária de imagens, distinguindo cães e gatos com alta precisão usando Redes Neurais Convolucionais (CNNs) em Python. Inclui scripts para organização automática do dataset, pré-processamento, modelagem, treinamento, avaliação e documentação técnica em LaTeX.

---

## 🚀 Como Executar

### 1. Preparação do Dataset

1. Baixe manualmente o arquivo `dogs-vs-cats.zip` do Kaggle:
   [https://www.kaggle.com/c/dogs-vs-cats/data](https://www.kaggle.com/c/dogs-vs-cats/data)

2. Coloque o arquivo na raiz do projeto.

3. Execute o script de preparação:

`bash download_data.sh`

O script irá criar as pastas, mover e extrair os arquivos zip, e organizar as imagens em `dataset/train/` e `dataset/test1/`.

### 2. Instale as Dependências

pip install -r requirements.txt

### 3. Treinamento e Avaliação

Execute o script principal para treinar e avaliar o modelo:

`python main.py`

- O script divide os dados em treino, validação e teste (70-15-15), faz aumento de dados e normalização, treina a CNN e salva métricas, gráficos e o modelo treinado.

---

## 🧠 Principais Recursos

- **CNN otimizada:** 3 blocos Conv2D+MaxPooling, camada densa com dropout, saída sigmoide.
- **Aumento de dados:** Rotação, deslocamento, shear, zoom e flip horizontal.
- **Regularização:** EarlyStopping e ModelCheckpoint.
- **Avaliação completa:** Acurácia, precisão, recall, matriz de confusão, curvas de aprendizado.
- **Organização automática de dados:** Script Bash para extração e organização do dataset Kaggle.

---

## 📊 Resultados Esperados

- **Acurácia no teste:** ~86.67%
- **Acurácia balanceada:** ~86.67%
- **Precisão/Recall (Gatos):** ~87% / ~86%
- **Precisão/Recall (Cães):** ~86% / ~88%

---
