#!/bin/bash

echo "=============================================="
echo "  Dogs vs Cats - Preparação do dataset"
echo "=============================================="
echo ""
echo "Este script irá organizar automaticamente o dataset Dogs vs Cats."
echo ""
echo "PASSO 1: Baixe manualmente o arquivo 'dogs-vs-cats.zip' do Kaggle."
echo "         (copie e cole o link abaixo em seu navegador):"
echo ""
echo "https://www.kaggle.com/c/dogs-vs-cats/data"
echo ""
echo "PASSO 2: Coloque o arquivo 'dogs-vs-cats.zip' no diretório do projeto."
echo ""
read -p "Quando o arquivo estiver no diretório, pressione ENTER para começar..."

ORIG_ZIP="dogs-vs-cats.zip"
DATASET_DIR="dataset"

# 1. Cria a pasta dataset/ se não existir
if [ ! -d "$DATASET_DIR" ]; then
    mkdir "$DATASET_DIR"
    echo "Pasta '$DATASET_DIR/' criada."
fi

# 2. Move o dogs-vs-cats.zip para dataset/ se estiver no diretório do projeto
if [ -f "$ORIG_ZIP" ]; then
    mv "$ORIG_ZIP" "$DATASET_DIR/"
    echo "Arquivo '$ORIG_ZIP' movido para '$DATASET_DIR/'."
fi

# 3. Verifica se dogs-vs-cats.zip está em dataset/
if [ ! -f "$DATASET_DIR/$ORIG_ZIP" ]; then
    echo "ERRO: '$ORIG_ZIP' não encontrado em '$DATASET_DIR/'."
    echo "Repita o PASSO 1 e PASSO 2 corretamente."
    exit 1
fi

cd "$DATASET_DIR"

echo ""
echo "Próxima etapa: extração do arquivo dogs-vs-cats.zip."
read -p "Pressione ENTER para continuar..."

# 4. Extrai dogs-vs-cats.zip (apenas se ainda não foi extraído)
if [ ! -f "train.zip" ] || [ ! -f "test1.zip" ]; then
    echo "Extraindo dogs-vs-cats.zip..."
    unzip -q -o dogs-vs-cats.zip
else
    echo "train.zip e test1.zip já existem, pulando extração de dogs-vs-cats.zip."
fi

echo ""
echo "Próxima etapa: extração do arquivo train.zip."
read -p "Pressione ENTER para continuar..."

# 5. Extrai train.zip sem criar subpasta extra
if [ -f "train.zip" ]; then
    mkdir -p train
    echo "Extraindo train.zip diretamente para a pasta train/..."
    unzip -q -j train.zip -d train
    echo "Extração de train.zip concluída."
else
    echo "ERRO: train.zip não encontrado após extração de dogs-vs-cats.zip."
    exit 1
fi

echo ""
echo "Próxima etapa: extração do arquivo test1.zip."
read -p "Pressione ENTER para continuar..."

# 6. Extrai test1.zip sem criar subpasta extra
if [ -f "test1.zip" ]; then
    mkdir -p test1
    echo "Extraindo test1.zip diretamente para a pasta test1/..."
    unzip -q -j test1.zip -d test1
    echo "Extração de test1.zip concluída."
else
    echo "ERRO: test1.zip não encontrado após extração de dogs-vs-cats.zip."
    exit 1
fi

# 7. Opção de exclusão dos arquivos zip
echo ""
read -p "Deseja excluir os arquivos ZIP originais para liberar espaço em disco? (s/n): " RESP
if [[ "$RESP" =~ ^[Ss]$ ]]; then
    echo "Removendo arquivos ZIP..."
    rm -f dogs-vs-cats.zip train.zip test1.zip
    echo "Arquivos ZIP removidos."
else
    echo "Os arquivos ZIP foram mantidos."
fi

echo ""
echo "As imagens de treino estão em 'dataset/train/' e as de teste em 'dataset/test1/'."
