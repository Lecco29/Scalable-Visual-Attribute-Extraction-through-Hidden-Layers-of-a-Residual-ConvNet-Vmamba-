#!/usr/bin/env python3
# experimento vmamba para classificacao de atributos de roupas

import os
import sys
import torch
import numpy as np
import pandas as pd
from datetime import datetime
from PIL import Image
from tqdm import tqdm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
from torchvision import transforms

RAIZ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, RAIZ)
sys.path.insert(0, os.path.join(RAIZ, 'models'))


def carregarDataset(caminhoCsv, pastaImagens):
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


def extrairFeatures(extrator, imagens, dispositivo, tamanhoBatch=64):
    todasFeatures = {f'stage{i+1}': [] for i in range(4)}
    nBatches = (len(imagens) + tamanhoBatch - 1) // tamanhoBatch
    
    for i in tqdm(range(nBatches), desc="Extraindo"):
        inicio = i * tamanhoBatch
        fim = min((i + 1) * tamanhoBatch, len(imagens))
        batch = imagens[inicio:fim].to(dispositivo)
        
        features = extrator.extrairFeatures(batch, aplicarGAP=True)
        
        for estagio, feat in features.items():
            todasFeatures[estagio].append(feat.numpy())
    
    for estagio in todasFeatures:
        todasFeatures[estagio] = np.vstack(todasFeatures[estagio])
    
    return todasFeatures


def avaliarKNN(features, labels, k=5, nfolds=5):
    encoder = LabelEncoder()
    y = encoder.fit_transform(labels)
    
    resultados = {}
    
    for estagio, X in features.items():
        knn = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
        acuracias = cross_val_score(knn, X, y, cv=nfolds, scoring='accuracy')
        
        knn.fit(X, y)
        predicao = knn.predict(X)
        f1 = f1_score(y, predicao, average='weighted')
        
        resultados[estagio] = {
            'accuracy_mean': acuracias.mean() * 100,
            'accuracy_std': acuracias.std() * 100,
            'f1_score': f1 * 100,
            'dim': X.shape[1]
        }
    
    return resultados


def rodarExperimento(nome, caminhoCsv, pastaImagens, extrator, dispositivo):
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
    
    # hardware
    if torch.cuda.is_available():
        dispositivo = 'cuda'
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        dispositivo = 'cpu'
        print("Usando CPU")
    
    # carrega vmamba
    print("\nCarregando modelo...")
    from carregadorVMamba import criarExtrator
    extrator = criarExtrator(dispositivo=dispositivo)
    
    pastaData = os.path.join(RAIZ, 'data')
    
    # experimentos
    resultadosCor = rodarExperimento(
        "COR",
        os.path.join(pastaData, 'labels_color.csv'),
        os.path.join(pastaData, 'images', 'color'),
        extrator, dispositivo
    )
    
    resultadosTextura = rodarExperimento(
        "TEXTURA",
        os.path.join(pastaData, 'labels_texture.csv'),
        os.path.join(pastaData, 'images', 'texture'),
        extrator, dispositivo
    )
    
    # salva csv
    for nome, resultados in [('color', resultadosCor), ('texture', resultadosTextura)]:
        df = pd.DataFrame([{'stage': e, **d} for e, d in resultados.items()])
        df.to_csv(os.path.join(RAIZ, f'results_{nome}_final.csv'), index=False)


if __name__ == '__main__':
    main()
