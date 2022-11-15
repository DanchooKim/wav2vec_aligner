## **wav2vec_aligner**

### 요약
> 입력한 한국어 음성으로 부터. 입력한 텍스트에 해당하는 음성 만을 뽑아내는 도구입니다.
> 음성만 입력한 경우, 묵음 구간을 제거하는 ctc기반 offline-epd로 동작합니다.

### 구축 이유
> 타임 스탬프가 없는 대본만 있는 상황에서, TTS용 학습 데이터 구축을 위해 일일이 눈으로 봐가면서 음성을 자르기가 힘들어서 만들었습니다.

### 사용 방법
> code의 main을 참고하시면 됩니다.

### 요구사항
> pytorch 1.10 이상
> tensorflow-cpu (gpu 버전이면 안됩니다)
> jamo
> hangul_utils
> transformers

### 현재버그
> colab에서 작동이 안됩니다. 이유는 모르겠습니다;

### Reference
>  https://pytorch.org/tutorials/intermediate/forced_alignment_with_torchaudio_tutorial.html
>  https://huggingface.co/hyyoka/wav2vec2-xlsr-korean-senior
