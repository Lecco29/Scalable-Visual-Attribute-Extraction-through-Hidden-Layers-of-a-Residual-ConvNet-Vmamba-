#!/usr/bin/env python3
# experimento vmamba - classificacao de atributos de roupas
# usando folds pre-definidos do protocolo

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
from avaliacaoKNN import avaliarKNNComFolds


def carregarImagensDeFolds(pastaProtocolo, tipoDataset, pastaImagens):
    # carrega todas as imagens baseado nos arquivos de fold
    # retorna imagens, labels e nomes dos arquivos
    
    if tipoDataset == 'color':
        pastaFolds = os.path.join(pastaProtocolo, 'folds_color', 'folds')
        prefixo = 'color'
    else:
        pastaFolds = os.path.join(pastaProtocolo, 'folds_texture', 'folds')
        prefixo = 'texture'
    
    # coleta todos os arquivos unicos de todos os folds
    todosArquivos = {}  # nomeOriginal -> (classe, caminhoLocal)
    
    for numFold in range(1, 6):
        for tipo in ['train', 'val', 'test']:
            arquivo = os.path.join(pastaFolds, f'fold{numFold}-{tipo}.txt')
            
            with open(arquivo, 'r') as arquivoLeitura:
                for linha in arquivoLeitura:
                    partes = linha.strip().split(';')
                    if len(partes) >= 3:
                        nomeClasse = partes[1]
                        caminhoOriginal = partes[2]
                        nomeOriginal = os.path.basename(caminhoOriginal)  # ex: 6996.jpg
                        
                        # monta nome local: color_amarillo_6996.jpg ou texture_polka_123.jpg
                        nomeBase, extensao = os.path.splitext(nomeOriginal)
                        nomeLocal = f"{prefixo}_{nomeClasse}_{nomeBase}{extensao}"
                        caminhoLocal = os.path.join(pastaImagens, nomeLocal)
                        
                        if nomeOriginal not in todosArquivos:
                            todosArquivos[nomeOriginal] = (nomeClasse, caminhoLocal, nomeOriginal)
    
    # carrega as imagens
    transformacao = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    listaImagens = []
    listaRotulos = []
    listaNomes = []
    
    for nomeOriginal, (classe, caminhoLocal, nomeArquivo) in todosArquivos.items():
        if os.path.exists(caminhoLocal):
            try:
                imagem = Image.open(caminhoLocal).convert('RGB')
                listaImagens.append(transformacao(imagem))
                listaRotulos.append(classe)
                listaNomes.append(nomeOriginal)  # guarda nome original do fold
            except Exception as erro:
                print(f"Erro ao carregar {caminhoLocal}: {erro}")
        else:
            print(f"Nao encontrado: {caminhoLocal}")
    
    if len(listaImagens) == 0:
        raise RuntimeError(f"Nenhuma imagem encontrada em {pastaImagens}")
    
    return torch.stack(listaImagens), listaRotulos, listaNomes


def rodarExperimento(nome, tipoDataset, pastaProtocolo, pastaImagens, extrator, dispositivo):
    # roda experimento completo usando folds pre-definidos
    
    print(f"\n--- {nome} ---")
    
    print("Carregando imagens dos folds...")
    imagens, rotulos, nomes = carregarImagensDeFolds(pastaProtocolo, tipoDataset, pastaImagens)
    print(f"Total: {len(imagens)} imagens, {len(set(rotulos))} classes")
    
    print("Extraindo features...")
    features = extrairFeatures(extrator, imagens, dispositivo)
    
    print("Avaliando com kNN (usando folds pre-definidos)...")
    resultados = avaliarKNNComFolds(features, rotulos, nomes, pastaProtocolo, tipoDataset)
    
    print(f"\nResultados {nome}:")
    for estagio in ['stage1', 'stage2', 'stage3', 'stage4']:
        resultadoEstagio = resultados[estagio]
        print(f"  {estagio}: {resultadoEstagio['accuracy_mean']:.2f}% (+/- {resultadoEstagio['accuracy_std']:.2f})")
    
    return resultados


def main():
    print("Experimento VMamba - Atributos de Roupas")
    print("Usando protocolo de folds pre-definidos")
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
    pastaProtocolo = os.path.join(pastaData, 'Protocolo')
    
    # experimento de cor
    resultadosCor = rodarExperimento(
        "COR",
        'color',
        pastaProtocolo,
        os.path.join(pastaData, 'images', 'color'),
        extrator, dispositivo
    )
    
    # experimento de textura
    resultadosTextura = rodarExperimento(
        "TEXTURA",
        'texture',
        pastaProtocolo,
        os.path.join(pastaData, 'images', 'texture'),
        extrator, dispositivo
    )
    
    # salva resultados em csv
    for nomeExperimento, resultadosExperimento in [('color', resultadosCor), ('texture', resultadosTextura)]:
        dadosFrame = pd.DataFrame([{'stage': estagio, **dados} for estagio, dados in resultadosExperimento.items()])
        dadosFrame.to_csv(os.path.join(RAIZ, f'results_{nomeExperimento}_final.csv'), index=False)
    
    print(f"\nFim: {datetime.now().strftime('%H:%M:%S')}")


if __name__ == '__main__':
    main()
