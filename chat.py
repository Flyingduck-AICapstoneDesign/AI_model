import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def main():
    # 1. 모델과 저장된 가중치 경로 설정
    MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
    LORA_DIR = "./baseline_outputs/final_law_lora"  # 방금 학습이 끝난 우리의 결과물!

    print("▶ 1. 토케나이저 및 베이스 모델 로드 중 (잠시만 기다려주세요)...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

    # 뼈대가 되는 베이스 모델 불러오기
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        device_map="auto",
        trust_remote_code=True
    )

    print("▶ 2. 학습된 LoRA 가중치(법률 지식) 뇌에 이식 중...")
    model = PeftModel.from_pretrained(base_model, LORA_DIR)
    model.eval()  # 평가(추론) 모드로 전환

    print("\n" + "=" * 50)
    print("⚖️ 민사소송 전문 챗봇이 준비되었습니다! (종료하려면 'q' 입력)")
    print("=" * 50 + "\n")

    # 3. 무한 루프로 채팅하기
    while True:
        user_input = input("👤 서현: ")
        if user_input.lower() == 'q':
            print("대화를 종료합니다. 수고하셨습니다!")
            break

        # 우리가 학습시켰던 ChatML 포맷 그대로 질문 포장하기
        prompt = (
            "<|im_start|>system\n당신은 대한민국의 민사소송 절차와 법률 지식을 안내하는 전문 법률 챗봇입니다.<|im_end|>\n"
            f"<|im_start|>user\n{user_input}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        # 답변 생성
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,  # 최대 생성 길이
            temperature=0.7,  # 창의성 (너무 높으면 헛소리, 너무 낮으면 로봇 같음)
            top_p=0.9,
            repetition_penalty=1.1,  # 반복 말하기 방지
            pad_token_id=tokenizer.eos_token_id
        )

        # 질문 부분(input)은 잘라내고, 모델이 새로 생성한 대답(output)만 가져오기
        input_length = inputs.input_ids.shape[1]
        response = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)

        print(f"\n🤖 챗봇: {response}\n")
        print("-" * 50)


if __name__ == "__main__":
    main()