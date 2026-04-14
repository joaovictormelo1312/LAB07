# Laboratório 07: Especialização de LLMs com LoRA e QLoRA

Este repositório contém o pipeline completo para o fine-tuning de um modelo Llama 2 7B no domínio de **Consultoria Financeira Pessoal e Investimentos**, utilizando técnicas de eficiência de parâmetros.

## Estrutura do Projeto

- `generate_dataset.py`: Script para geração de dados sintéticos via API OpenAI (conforme requisito).
- `fine_tune.py`: Script principal de treinamento usando QLoRA, LoRA e SFTTrainer.
- `dataset_train.jsonl`: Conjunto de treino (45 exemplos).
- `dataset_test.jsonl`: Conjunto de teste (6 exemplos).
- `create_data_helper.py`: Script auxiliar usado para gerar os dados iniciais.

## Configurações Implementadas (Requisitos do Lab)

- **Quantização (Passo 2)**: 
    - BitsAndBytes: 4-bits.
    - Tipo: `nf4`.
    - Compute Dtype: `float16`.
- **Arquitetura LoRA (Passo 3)**:
    - Rank (r): 64.
    - Alpha (alpha): 16.
    - Dropout: 0.1.
    - Tarefa: `CAUSAL_LM`.
- **Pipeline de Treinamento (Passo 4)**:
    - Otimizador: `paged_adamw_32bit`.
    - Scheduler: `cosine`.
    - Warmup Ratio: 0.03.

## Como Executar

### 1. Requisitos
- Python 3.9+
- GPU NVIDIA (mínimo 16GB VRAM recomendado para Llama 2 7B)
- Token do Hugging Face com acesso ao Llama 2.

### 2. Instalação
```bash
pip install torch transformers peft bitsandbytes trl accelerate datasets scikit-learn
```

### 3. Login no Hugging Face
```bash
huggingface-cli login
```

### 4. Executar Fine-Tuning
```bash
python fine_tune.py
```

## Nota de Integridade (Obrigatória)

> [!IMPORTANT]
> **Partes geradas/complementadas com IA, revisadas por João Victor Melo**

---
Desenvolvido para fins acadêmicos.
