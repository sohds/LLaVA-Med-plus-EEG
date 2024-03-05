# 🏥 LLaVA-Med-plus-EEG

Daiv 2024 Winter Chllenge Medical AI LLM Team <b>'셋둘하나'</b>

## Team members

| <img src="./profiles/다운 깃허브 프로필.jfif" width="100"> | <img src="./profiles/서연 깃허브 프로필.png" width="100"> | <img src="./profiles/예현 깃허브 프로필.png" width="100"> | <img src="./profiles/준영 깃허브 프로필.png" width="100"> |
| :--------------------------------------------------------: | :-------------------------------------------------------: | :-------------------------------------------------------: | :-------------------------------------------------------: |
|    <a href="https://github.com/drawcodeboy">권다운</a>     |       <a href="https://github.com/sohds">오서연</a>       |      <a href="https://github.com/kyean22">고예현</a>      |  <a href="https://github.com/crinex">박준영 (Mentor)</a>  |

## Introduction

## Baseline Model

### 필요 작업

1. LLaMA에서 weight를 제공받은 뒤, 모델에 적용을 시킨다.
2. LLaVA-Med에서 제공하는 Delta Weight를 모델에 적용시켜 사용할 수 있는 모델 상태를 유지한다.
3. 큰 RAM이 필요한 작업

|           Type           |                                                         Link                                                         |
| :----------------------: | :------------------------------------------------------------------------------------------------------------------: |
|        Repository        |                                 [LLaVA-Med](https://github.com/microsoft/LLaVA-Med)                                  |
| Fine-tuned Delta Weights | [Slake1.0-9epoch_delta.zip](https://hanoverprod.z21.web.core.windows.net/med_llava/models/Slake1.0-9epoch_delta.zip) |
|        Base Model        |                  [HuggingFace_LLaMA](https://huggingface.co/docs/transformers/main/model_doc/llama)                  |
