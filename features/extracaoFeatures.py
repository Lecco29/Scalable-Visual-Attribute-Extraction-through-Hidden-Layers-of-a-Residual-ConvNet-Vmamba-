# extracao de features do vmamba

import numpy as np
import torch
from tqdm import tqdm


def extrairFeatures(extrator, imagens, dispositivo, tamanhoBatch=64):
    # extrai features de todas as imagens usando o vmamba
    # retorna dicionario com features de cada estagio
    
    todasFeatures = {f'stage{i+1}': [] for i in range(4)}
    nBatches = (len(imagens) + tamanhoBatch - 1) // tamanhoBatch
    
    for i in tqdm(range(nBatches), desc="Extraindo"):
        inicio = i * tamanhoBatch
        fim = min((i + 1) * tamanhoBatch, len(imagens))
        batch = imagens[inicio:fim].to(dispositivo)
        
        # extrai features com global average pooling
        features = extrator.extrairFeatures(batch, aplicarGAP=True)
        
        for estagio, feat in features.items():
            todasFeatures[estagio].append(feat.numpy())
    
    # junta todos os batches
    for estagio in todasFeatures:
        todasFeatures[estagio] = np.vstack(todasFeatures[estagio])
    
    return todasFeatures

