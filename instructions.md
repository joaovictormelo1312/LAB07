# Instruções de Execução: Laboratório 07

Este guia detalha os passos necessários para configurar o ambiente e rodar o pipeline de fine-tuning.

## 1. Pré-requisitos de Hardware
O treinamento do Llama 2 7B, mesmo com QLoRA, exige uma GPU com arquitetura moderna (NVIDIA):
- **VRAM Recomendada**: 16GB (ex: RTX 3090, 4080, A100, T4 no Colab).
- **VRAM Mínima**: 12GB (pode ser necessário reduzir o `per_device_train_batch_size` para 1 ou 2 no `fine_tune.py`).

## 2. Configuração do Ambiente
Recomenda-se o uso de um ambiente virtual (venv ou conda).

```bash
# Criar ambiente virtual
python -m venv venv

# Ativar no Windows
.\venv\Scripts\activate

# Instalar dependências necessárias
pip install torch transformers peft bitsandbytes trl accelerate datasets scikit-learn
```

## 3. Acesso ao Modelo Gated (Llama 2)
O Llama 2 não pode ser baixado sem autorização da Meta. 
1. Solicite acesso no [Hugging Face](https://huggingface.co/meta-llama/Llama-2-7b-hf).
2. Crie um **Access Token** em: *Settings -> Access Tokens* no seu perfil do Hugging Face.
3. No terminal, faça o login:
   ```bash
   huggingface-cli login
   ```
   Cole o seu token quando solicitado.

## 4. Verificação dos Dados
Certifique-se de que os arquivos `dataset_train.jsonl` e `dataset_test.jsonl` estão na mesma pasta que o script `fine_tune.py`. Eles já foram gerados por mim para você.

## 5. Executando o Fine-Tuning
Para iniciar o treinamento, execute:

```bash
python fine_tune.py
```

### O que acontece durante a execução:
- O modelo será carregado em 4-bits (ocupando ~5-6GB de VRAM inicial).
- O dataset será carregado e formatado.
- O treinamento iniciará. Você verá logs de perda (`loss`) a cada 25 passos.
- Ao final, uma pasta chamada `llama2-specialized-finance` será criada contendo os **adapters** (pesos treinados).

## 6. Solução de Problemas (OOM - Out of Memory)
Se você receber um erro de "Out of Memory":
1. Abra `fine_tune.py`.
2. Altere `per_device_train_batch_size=4` para `per_device_train_batch_size=1`.
3. Adicione `gradient_accumulation_steps=4` para manter o batch size efetivo.

## 7. Critérios de Avaliação (Lembrete)
Após rodar com sucesso:
1. Suba os arquivos para o GitHub (o `.gitignore` impedirá que você suba os pesos pesados).
2. Adicione seu nome no `README.md`.
3. Crie uma tag `v1.0` no GitHub.
