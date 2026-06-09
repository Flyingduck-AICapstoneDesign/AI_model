from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import uvicorn

# --- [추가] 비동기 격리 처리를 위한 라이브러리 ---
import asyncio
from concurrent.futures import ThreadPoolExecutor

# 1. FastAPI 앱 생성
app = FastAPI(title="민사소송 AI 챗봇 서버")

# --- [추가] AI 연산만을 담당할 전역 스레드 풀 생성 (동시 요청 방어) ---
executor = ThreadPoolExecutor(max_workers=2)

# 2. JSON 키 구조 정의
class ChatRequest(BaseModel):
    session_id: str = "default_user"
    question: str  # 프론트에서 보내는 질문

class ChatResponse(BaseModel):
    status: str
    answer: str    # 서버가 프론트로 보낼 답변
    error_msg: str = ""

# 3. 전역 변수로 모델과 토케나이저 선언
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
LORA_DIR = "./baseline_outputs/final_law_lora"
tokenizer = None
model = None

# 서버 켜질 때 AI 뇌 로딩하기
@app.on_event("startup")
def load_model():
    global tokenizer, model
    print(":arrow_forward: AI 모델을 서버 메모리에 적재 중 (약 1~2분 소요)")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    model = PeftModel.from_pretrained(base_model, LORA_DIR)
    model.eval()
    print(":arrow_forward: AI 서버 가동 준비 완료")

# 헬스체크 엔드포인트
@app.get("/health")
def health_check():
    return {"status": "ok", "message": "민사소송 AI 서버 정상 작동 중"}


# --- [수정] 외부 통신을 끊지 않기 위해 실제 추론만 따로 수행할 순수 동기 함수 분리 ---
def run_model_inference(prompt):
    global tokenizer, model
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1024,
            temperature=0.3,
            top_p=0.85,
            repetition_penalty=1.2,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    input_length = inputs.input_ids.shape[1]
    response_text = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
    return response_text


# --- [수정] 메인 챗봇 엔드포인트를 비동기(async def)로 변경 ---
@app.post("/api/chat", response_model=ChatResponse)
async def generate_chat(request: ChatRequest):

    print(f"[입력 질문]: {request.question}")

    try:
        prompt = (
            "<|im_start|>system\n당신은 대한민국의 민사소송 절차와 법률 지식을 안내하는 전문 법률 챗봇입니다.<|im_end|>\n"
            f"<|im_start|>user\n{request.question}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

        # --- [핵심 수정] AI 연산이 도는 동안 대기(await)하되, 서버 통신 스레드는 계속 ngrok과 소통하게 만듦 ---
        loop = asyncio.get_event_loop()
        response_text = await loop.run_in_executor(executor, run_model_inference, prompt)

        return ChatResponse(status="success", answer=response_text)

    except Exception as e:
        print(f"Error: {e}")
        return ChatResponse(status="error", answer="서버 내부 오류가 발생", error_msg=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)