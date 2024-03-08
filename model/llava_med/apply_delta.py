'''
출처: https://github.com/microsoft/LLaVA-Med/blob/main/llava/model/apply_delta.py

LLaVA-Med의 delta weight를 적용할 때, 필요한 script 파일
더 분석 필요하다고 생각함. AutoModelForCausalLM
'''

import argparse
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import AutoConfig, StoppingCriteria
from llava import LlavaLlamaForCausalLM

def apply_delta(base_model_path, target_model_path, delta_path):
    
    # LLaMA 모델을 가져오는 과정
    # Request 요청한 모델이 이 모델이다.
    print("Loading base model")
    base = AutoModelForCausalLM.from_pretrained(
        base_model_path, torch_dtype=torch.float16, low_cpu_mem_usage=True)

    # Delta Weight 불러오기
    # 다운받았던 Delta weight 파일을 불러온다.
    print("Loading delta")
    delta = LlavaLlamaForCausalLM.from_pretrained(delta_path, torch_dtype=torch.float16, low_cpu_mem_usage=True)
    delta_tokenizer = AutoTokenizer.from_pretrained(delta_path)

    # LLaMA + Delta Weight -> LLaVA-Med (fine-tuned by SLAKE)
    print("Applying delta")
    for name, param in tqdm(delta.state_dict().items(), desc="Applying delta"):
        if name not in base.state_dict():
            # Delta Weight에는 LLaMA의 weight만 존재하는 것은 아니다.
            # 그래서 LLaMA weight 이외의 다른 weight를 걸러내기 위한 조건식이다.
            # ['model.mm_projector.weight', 'model.mm_projector.bias']는 LLaMA에서 쓰이지는 않으나
            # LLaVA-Med의 관점에서 필요한 Layer이다.
            # 추측으로는 mm_projector는 CLIP과 LLaMA를 연결해주는 역할을 하는 듯 하다.
            # 결론적으로, LLaMA와 두 모델을 연결하는 Projection Layer가 아니라면 Assertion Error를 발생시킨다.
            assert name in ['model.mm_projector.weight', 'model.mm_projector.bias'], f'{name} not in base model'
            continue
        if param.data.shape == base.state_dict()[name].shape:
            # 동일한 Parameter인지 확인 후에 Base의 Parameter를 Delta에 더한다.
            # 여기가 Delta Weight를 제대로 사용하는 부분
            param.data += base.state_dict()[name]
        else:
            # Token Embedding, Head 부분이면 시작과 끝이라는 것을 유추할 수 있다.
            # 해당 Layer들은 시작과 끝의 Shape이 살짝 달라서 따로 조건을 분기하여 합치는
            # 작업을 해준 것으로 보인다.
            assert name in ['model.embed_tokens.weight', 'lm_head.weight'], \
                f'{name} dimension mismatch: {param.data.shape} vs {base.state_dict()[name].shape}'
            bparam = base.state_dict()[name]
            param.data[:bparam.shape[0], :bparam.shape[1]] += bparam

    print("Saving target model")
    # 저장을 함으로써 Hugging Face의 라이브러리처럼 .from_pretrained로 쓸 수 있다.
    delta.save_pretrained(target_model_path)
    delta_tokenizer.save_pretrained(target_model_path)

    print("Done!")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model-path", type=str, required=True)
    parser.add_argument("--target-model-path", type=str, required=True)
    parser.add_argument("--delta-path", type=str, required=True)

    args = parser.parse_args()

    apply_delta(args.base_model_path, args.target_model_path, args.delta_path)