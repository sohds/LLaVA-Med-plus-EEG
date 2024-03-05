'''
출처: https://github.com/microsoft/LLaVA-Med/blob/main/llava/model/apply_delta.py

LLaVA-Med의 delta weight를 적용할 때, 필요한 script 파일

아직 실행 조건이 안 되어 실행 해보지 않았으며, 정확한 코드 분석이 필요.
또한, llava는 아직 다운받지 않음.
'''

import argparse
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import AutoConfig, StoppingCriteria
from llava import LlavaLlamaForCausalLM

def auto_upgrade(config):
    cfg = AutoConfig.from_pretrained(config)
    if 'llava' in config and cfg.model_type != 'llava':
        print("You are using newer LLaVA code base, while the checkpoint of v0 is from older code base.")
        print("You must upgrade the checkpoint to the new code base (this can be done automatically).")
        confirm = input("Please confirm that you want to upgrade the checkpoint. [Y/N]")
        if confirm.lower() in ["y", "yes"]:
            print("Upgrading checkpoint...")
            assert len(cfg.architectures) == 1
            setattr(cfg.__class__, "model_type", "llava")
            cfg.architectures[0] = 'LlavaLlamaForCausalLM'
            cfg.save_pretrained(config)
            print("Checkpoint upgraded.")
        else:
            print("Checkpoint upgrade aborted.")
            exit(1)

def apply_delta(base_model_path, target_model_path, delta_path):
    print("Loading base model")
    base = AutoModelForCausalLM.from_pretrained(
        base_model_path, torch_dtype=torch.float16, low_cpu_mem_usage=True)

    print("Loading delta")
    delta = LlavaLlamaForCausalLM.from_pretrained(delta_path, torch_dtype=torch.float16, low_cpu_mem_usage=True)
    delta_tokenizer = AutoTokenizer.from_pretrained(delta_path)

    print("Applying delta")
    for name, param in tqdm(delta.state_dict().items(), desc="Applying delta"):
        if name not in base.state_dict():
            assert name in ['model.mm_projector.weight', 'model.mm_projector.bias'], f'{name} not in base model'
            continue
        if param.data.shape == base.state_dict()[name].shape:
            param.data += base.state_dict()[name]
        else:
            assert name in ['model.embed_tokens.weight', 'lm_head.weight'], \
                f'{name} dimension mismatch: {param.data.shape} vs {base.state_dict()[name].shape}'
            bparam = base.state_dict()[name]
            param.data[:bparam.shape[0], :bparam.shape[1]] += bparam

    print("Saving target model")
    delta.save_pretrained(target_model_path)
    delta_tokenizer.save_pretrained(target_model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model-path", type=str, required=True)
    parser.add_argument("--target-model-path", type=str, required=True)
    parser.add_argument("--delta-path", type=str, required=True)

    args = parser.parse_args()

    apply_delta(args.base_model_path, args.target_model_path, args.delta_path)