#!/usr/bin/env python3
# experimento vmamba - classificacao de atributos de roupas

import os
import sys
import torch
import pandas as pd
from datetime import datetime
from PIL import Image
from torchvision import transforms

# adiciona os caminhos
RAIZ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, RAIZ)
sys.path.insert(0, os.path.join(RAIZ, 'models'))
sys.path.insert(0, os.path.join(RAIZ, 'features'))
sys.path.insert(0, os.path.join(RAIZ, 'analysis'))

# imports dos modulos do projeto
from carregadorVMamba import criarExtrator
from extracaoFeatures import extrairFeatures
from avaliacaoKNN import avaliarKNN


def carregarDataset(caminhoCsv, pastaImagens):
    # carrega imagens e labels de um csv
    
    df = pd.read_csv(caminhoCsv)
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    imagens = []
    labels = []
    
    for idx, linha in df.iterrows():
        caminhoImg = os.path.join(pastaImagens, linha['image_name'])
        if os.path.exists(caminhoImg):
            try:
                img = Image.open(caminhoImg).convert('RGB')
                imagens.append(transform(img))
                labels.append(linha['label'])
            except:
                pass
    
    return torch.stack(imagens), labels


def rodarExperimento(nome, caminhoCsv, pastaImagens, extrator, dispositivo):
    # roda experimento completo: carrega dados, extrai features, avalia
    
    print(f"\n--- {nome} ---")
    
    print("Carregando imagens...")
    imagens, labels = carregarDataset(caminhoCsv, pastaImagens)
    print(f"Total: {len(imagens)} imagens, {len(set(labels))} classes")
    
    print("Extraindo features...")
    features = extrairFeatures(extrator, imagens, dispositivo)
    
    print("Avaliando com kNN...")
    resultados = avaliarKNN(features, labels)
    
    print(f"\nResultados {nome}:")
    for estagio in ['stage1', 'stage2', 'stage3', 'stage4']:
        r = resultados[estagio]
        print(f"  {estagio}: {r['accuracy_mean']:.2f}% (+/- {r['accuracy_std']:.2f})")
    
    return resultados


def main():
    print("Experimento VMamba - Atributos de Roupas")
    print(f"Inicio: {datetime.now().strftime('%H:%M:%S')}")
    
    # verifica hardware
    if torch.cuda.is_available():
        dispositivo = 'cuda'
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        dispositivo = 'cpu'
        print("Usando CPU")
    
    # carrega o modelo vmamba
    print("\nCarregando modelo...")
    extrator = criarExtrator(dispositivo=dispositivo)
    
    pastaData = os.path.join(RAIZ, 'data')
    
    # experimento de cor
    resultadosCor = rodarExperimento(
        "COR",
        os.path.join(pastaData, 'labels_color.csv'),
        os.path.join(pastaData, 'images', 'color'),
        extrator, dispositivo
    )
    
    # experimento de textura
    resultadosTextura = rodarExperimento(
        "TEXTURA",
        os.path.join(pastaData, 'labels_texture.csv'),
        os.path.join(pastaData, 'images', 'texture'),
        extrator, dispositivo
    )
    
    # salva resultados em csv
    for nome, resultados in [('color', resultadosCor), ('texture', resultadosTextura)]:
        df = pd.DataFrame([{'stage': e, **d} for e, d in resultados.items()])
        df.to_csv(os.path.join(RAIZ, f'results_{nome}_final.csv'), index=False)


if __name__ == '__main__':
    main()
