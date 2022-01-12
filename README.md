# 공개된 자연어 처리 모델을 활용한 한국어 폭력/비폭력 상황 인지 (2021년 인공지능 그랜드 챌린지 음성인지 트랙)

**본 과제는 [정보통신기획평가원](https://www.iitp.kr/main.it)의 지원하에 수행되었으며 미디어 상에서 발생할 수 있는 다양한 한국어 폭력/비폭력 상황을 인지를 위해 공개된 한국어 기반 사전학습 언어 모델(KoELECTRA, KcELECTRA 등)을 활용할 수 있는 소스 코드 제공합니다.**

>과제명 : [멀티모달 데이터 기반 한국어 폭력 상황 감지를 위한 AI 기술 개발](https://www.ntis.go.kr/project/pjtInfo.do?pjtId=1711126136&pageCode=TH_MYPJT_PJT_DTL)

>Contact : wlsghks4043@kw.ac.kr, skgusrb12@kw.ac.kr

## 1. Introduction

본 과제는 일상생활에서 발생하는 다양한 폭력 상황을 인지하기 위해 한국어 폭력 텍스트 데이터를 자체적으로 구축하고, 공개된 자연어 처리 기술을 활용할 수 있는 공개 SW 제공을 목표로 합니다. 과제 수행에서 정의한 4가지의 한국어 폭력 상황과 1가지의 한국어 비폭력 상황은 [2021년 인공지능 그랜드 챌린지](https://www.ai-challenge.kr/)에서 정의한 폭력 상황 협박, 갈취 또는 공갈, 괴롭힘(직장 내 괴롭힘, 직장 외 괴롭힘)과 비폭력 상황을 기준으로 하였으며, 아래의 표에 자세히 설명되어 있습니다.

<p align="center"><br><img src="https://user-images.githubusercontent.com/51149957/148895420-6f455f0c-5502-4ba9-acc0-60b65a53a92a.png"  width="700" height="350"></br></p align="center">
<p align="center"><b>< 4가지의 한국어 폭력 상황 정의 ></b><p align="center">

<p align="center"><br><img src="https://user-images.githubusercontent.com/51149957/148897007-841b5e9a-f1c2-4a41-9bef-a336fd5a3c77.png"  width="700" height="350"></br></p align="center">
<p align="center"><b>< 구축된 한국어 폭력 발화 텍스트 예시 ></b><p align="center">  

제공되는 공개 SW에서는 Huggingface에 공개된 4가지의 한국어 기반 사전 학습 언어 모델(KoELECTRA, KcELECTRA, tunib-ELECTRA, KLUE-RoBERTa)과 ETRI에서 공개된 KorBERT를 활용하여 훈련 및 테스트, 모델의 앙상블을 통한 테스트까지 활용할 수 있습니다. 해당 모델에 대한 자세한 내용은 하단의 Reference를 통해 확인할 수 있습니다.


## 2. Installation






## 3. Usage


### Training

```
# only training
python run_classifier.py --is_training True --is_test False --is_ensemble_test False
```
  
  
### Inference
```
# only test using single model
python run_classifier.py --is_training False --is_test True --is_ensemble_test False
  
# ensemble test using multiple model
python run_classifier.py --is_training False --is_test True --is_ensemble_test True --ensemble_comb_n <ENSEMBLE_MODEL_NUMBER>
```  

## 4. Results

정의한 각각의 5가지의 한국어 폭력/비폭력 상황에 대해 자체적으로 구축한 100개의 Test 데이터에 대해 성능을 Macro-F1 score를 통해 평가하였습니다.





## Citations

해당 코드를 연구용으로 활용한다면, 아래와 같이 인용을 해주세요.

```bibtex
@misc{Choi2021KoreanHateSpeechRecognition,
  author       = {Choi, Young-Seok and Kim, Jin-Hwan and Lee, Hyeon Kyu},
  title        = {KHSR, Korean Hate Speech Recognition in Social Media},
  publisher    = {GitHub},
  journal      = {GitHub repository},
  howpublished = {\url{https://github.com/NeuroAI-PI/AI-Grand-Challenge-2021}},
  year         = {2021}
}
```

## Reference

- [Huggingface Transformers](https://github.com/huggingface/transformers)
- [KoELECTRA](https://github.com/monologg/KoELECTRA)
- [KcElECTRA](https://github.com/Beomi/KcELECTRA)
- [tunib-ELECTRA](https://github.com/tunib-ai/tunib-electra)
- [KorBERT](https://aiopen.etri.re.kr/service_dataset.php)
- [klue-RoBERTa](https://github.com/KLUE-benchmark/KLUE)



