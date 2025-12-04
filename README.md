# VMamba - Classificação de Atributos de Roupas

Projeto de IC que usa o VMamba pra classificar cor e textura de roupas.

## O que faz

Extrai features de diferentes camadas do VMamba e classifica com kNN.

## Resultados

| Estágio | Cor | Textura |
|---------|-----|---------|
| stage1 | 94.00% | 86.80% |
| stage2 | 92.20% | 92.50% |
| stage3 | 74.90% | 89.70% |
| stage4 | 47.50% | 73.90% |

**Conclusão:** Camadas iniciais são melhores pra cor, intermediárias pra textura.

## Como rodar

1. Criar ambiente virtual:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

2. Baixar o modelo:
```bash
# baixar model.safetensors do HuggingFace:
# https://huggingface.co/saurabhati/VMamba_ImageNet_82.6
```

3. Clonar e compilar o VMamba:
```bash
git clone https://github.com/MzeroMiko/VMamba vmamba_repo
cd vmamba_repo/kernels/selective_scan
pip install .
```

4. Rodar:
```bash
python experimentoFinal.py
```

## Estrutura

```
project/
├── experimentoFinal.py    # script principal
├── models/                # carregador do vmamba
├── data/                  # imagens e labels
├── modeling_vmamba.py     # arquitetura do modelo
└── results_*.csv          # resultados
```

## Dataset

1000 imagens de cor + 1000 de textura (10 classes cada).

## Requisitos

- Python 3.10+
- GPU com CUDA (opcional mas recomendado)
- ~6GB de RAM

