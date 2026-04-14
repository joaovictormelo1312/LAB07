import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer

# ==============================================================================
# Laboratório 07: Fine-Tuning de LLMs com LoRA e QLoRA
# ==============================================================================

# 1. Configurações Iniciais
model_name = "meta-llama/Llama-2-7b-hf"
dataset_train_path = "dataset_train.jsonl"
dataset_test_path = "dataset_test.jsonl"
new_model_name = "llama2-specialized-finance"

# 2. Configuração de Quantização (QLoRA) - Requisito Passo 2
# nf4: NormalFloat 4-bit
# compute_dtype: float16
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=False,
)

# 3. Carregar Modelo Base e Tokenizer
print(f"Carregando o modelo {model_name}...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    use_auth_token=True,
)
model.config.use_cache = False
model.config.pretraining_tp = 1

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True,
    use_auth_token=True,
)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right" # Llama 2 costuma ser treinado com padding à direita

# 4. Configuração do LoRA - Requisito Passo 3
# Rank (r): 64
# Alpha: 16
# Dropout: 0.1
# Tarefa: CAUSAL_LM
peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
)

# 5. Carregar e Formatar o Dataset
def format_instruction(sample):
    return f"### Instrução:\n{sample['instruction']}\n\n### Resposta:\n{sample['response']}"

dataset = load_dataset("json", data_files={"train": dataset_train_path, "test": dataset_test_path})

# 6. Pipeline de Treinamento e Otimização - Requisito Passo 4
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,                # Número de épocas
    per_device_train_batch_size=1,      # Batch size reduzido para 8GB VRAM
    gradient_accumulation_steps=4,      # Acumula gradientes para batch efetivo maior
    optim="paged_adamw_32bit",          # Otimizador (Requisito)
    save_steps=25,
    logging_steps=25,
    learning_rate=2e-4,
    weight_decay=0.001,
    fp16=True,                          # Uso de float16
    bf16=False,
    max_grad_norm=0.3,
    max_steps=-1,
    warmup_ratio=0.03,                  # Warmup Ratio (Requisito)
    group_by_length=True,
    lr_scheduler_type="cosine",         # Scheduler (Requisito)
    report_to="tensorboard"
)

# Inicializar o SFTTrainer
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset["train"],
    peft_config=peft_config,
    dataset_text_field="instruction", # O SFTTrainer formatará usando essa chave se não houver template
    max_seq_length=512,
    tokenizer=tokenizer,
    args=training_args,
    packing=False,
)

# Iniciar o Treinamento
print("Iniciando o treinamento...")
trainer.train()

# 7. Salvar o Modelo Adaptador (Requisito)
print(f"Salvando o adapter em {new_model_name}...")
trainer.model.save_pretrained(new_model_name)

print("Fine-tuning concluído!")
