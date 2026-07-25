import os
import pandas as pd
import torch
from datasets import DatasetDict, Dataset
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model, TaskType
import ast  # 안전한 문자열 평가를 위한 모듈

# -------------------------------
# 경로 설정
MODEL_NAME = "/home/ubuntu/llama-3.2-Korean-Bllossom-AICA-5B"
DATA_DIR = "./processed_data"
OUTPUT_DIR = "./results"
SAVE_DIR = "/home/ubuntu/My_Project_ollama"
LOG_DIR = "./logs"

# -------------------------------
# GPU 확인 및 캐시 비우기
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()
print(f"Using device: {device}")

# -------------------------------
# 데이터셋 로드
def load_processed_datasets():
    print("Loading processed datasets...")
    train_data = pd.read_csv(os.path.join(DATA_DIR, "train_processed.csv"))
    val_data = pd.read_csv(os.path.join(DATA_DIR, "val_processed.csv"))
    test_data = pd.read_csv(os.path.join(DATA_DIR, "test_processed.csv"))

    # 문자열 데이터를 리스트로 변환
    for df in [train_data, val_data, test_data]:
        df["input_ids"] = df["input_ids"].apply(ast.literal_eval)
        df["attention_mask"] = df["attention_mask"].apply(ast.literal_eval)
        # labels를 input_ids와 동일하게 설정
        df["labels"] = df["input_ids"]

    print(f"Train dataset shape: {train_data.shape}")
    print(f"Validation dataset shape: {val_data.shape}")
    print(f"Test dataset shape: {test_data.shape}")
    return train_data, val_data, test_data


train_data, val_data, test_data = load_processed_datasets()

# -------------------------------
# 데이터셋 구조 확인 및 Hugging Face Dataset 변환
def convert_to_hf_dataset(data):
    dataset = Dataset.from_pandas(data)
    # 데이터 확인
    print(f"Sample from dataset: {dataset[0]}")
    return dataset

datasets = DatasetDict({
    "train": convert_to_hf_dataset(train_data),
    "validation": convert_to_hf_dataset(val_data),
    "test": convert_to_hf_dataset(test_data)
})

# -------------------------------
# 토크나이저 로드 및 데이터 Collator 설정
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
print("Tokenizer loaded successfully.")

data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    padding="max_length",
    max_length=128,
    return_tensors="pt"
)

# -------------------------------
# 모델 로드 및 LoRA 적용
print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto"
)
print("Model loaded successfully.")

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# -------------------------------
# 훈련 설정
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=16,
    gradient_accumulation_steps=4,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=1,
    logging_dir=LOG_DIR,
    logging_steps=50,
    load_best_model_at_end=True,
    bf16=True,
    metric_for_best_model="loss"
)

# -------------------------------
# Trainer 생성
print("Initializing Trainer...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=datasets["train"],
    eval_dataset=datasets["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator
)

# -------------------------------
# 학습 시작
def train_model():
    print("Starting training...")
    try:
        trainer.train()
        print("Training complete.")
    except Exception as e:
        print(f"Training failed with error: {e}")

train_model()

# -------------------------------
# 모델 저장
print("Saving model and tokenizer...")
os.makedirs(SAVE_DIR, exist_ok=True)
lora_model.save_pretrained(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)
print(f"Model and tokenizer saved to {SAVE_DIR}")

# -------------------------------
# 평가
print("Evaluating on Test Dataset...")
try:
    results = trainer.evaluate(datasets["test"])
    print("Evaluation results:", results)
except Exception as e:
    print(f"Evaluation failed with error: {e}")
