import os

# 보기 싫은 tokenizer 경고 메시지를 깔끔하게 꺼줍니다.
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling  # 생성형 AI 전용 콜레이터로 변경!
)
from peft import LoraConfig, TaskType
from trl import SFTTrainer, SFTConfig


def main():
    MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
    DATASET_PATH = "law_train_dataset.jsonl"
    OUTPUT_DIR = "./baseline_outputs"

    print("▶ 1. 토케나이저 및 데이터셋 로드 중...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    dataset = load_dataset("json", data_files=DATASET_PATH, split="train")

    print("▶ 2. 베이스 모델 로드 중 (bfloat16)...")
    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=compute_dtype,
        device_map="auto",
        trust_remote_code=True
    )

    print("▶ 3. LoRA 어댑터 설정 중...")
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )

    print("▶ 4. 훈련 하이퍼파라미터 및 SFT 설정 세팅 중...")
    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        num_train_epochs=5,
        learning_rate=2e-4,
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="no",
        lr_scheduler_type="cosine",
        warmup_steps=100,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        gradient_checkpointing=True,
        report_to="none",
        dataloader_num_workers=2,
        dataset_text_field="text",
        max_seq_length=1024
    )

    print("▶ 5. 트레이너 초기화 및 학습 준비 완료.")
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        args=training_args,
        tokenizer=tokenizer,
        # 입력 데이터에서 자동으로 Labels를 생성해 주는 Causal LM 전용 콜레이터
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
    )

    print("🚀 [START] 파인튜닝 학습을 시작합니다.")
    trainer.train()

    print("▶ 7. 학습 완료! 최종 LoRA 가중치 저장 중...")
    final_model_path = os.path.join(OUTPUT_DIR, "final_law_lora")
    trainer.model.save_pretrained(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    print(f"🎉 모든 과정이 끝났습니다! 모델이 저장되었습니다: {final_model_path}")


if __name__ == "__main__":
    main()