import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# GPU 메모리 초기화 (선택사항)
torch.cuda.empty_cache()

# 모델 및 토크나이저 로드 (양자화 없이)
print("모델 로드 중...")
model = AutoModelForCausalLM.from_pretrained(
    "./llama-3.2-Korean-Bllossom-AICA-5B",  # 모델 경로
    device_map="auto",  # GPU 사용 시 자동 분배
    torch_dtype=torch.float16  # FP16 정밀도 사용
)
tokenizer = AutoTokenizer.from_pretrained("./llama-3.2-Korean-Bllossom-AICA-5B")
print("모델 로드 완료!")

# AI 답변 생성 함수
def generate_answer(prompt: str, max_length: int = 50):
    # 입력값 토크나이징
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")  # GPU 사용

    # 모델 출력
    try:
        print("모델 출력 확인 중...")
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_length,
            temperature=0.7,
            do_sample=True,
            top_p=0.95,
            repetition_penalty=1.1
        )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response
    except Exception as e:
        print(f"오류 발생: {e}")
        return None

# 로컬에서 질문하고 답변 받기
if __name__ == "__main__":
    print("질문을 입력하세요 (종료하려면 'exit' 입력):")
    while True:
        user_input = input()
        if user_input.lower() == "exit":
            print("프로그램을 종료합니다.")
            break
        response = generate_answer(user_input)
        if response:
            print(f"AI의 답변: {response}")
        else:
            print("응답을 생성할 수 없습니다.")
