
# 타이로드 MoE 좌굴하중 예측 웹 프로그램

## 구성 파일

- `app.py`: Streamlit 웹 프로그램
- `requirements.txt`: Python 패키지
- `sample_batch.csv`: 일괄 예측 예시
- `tie_rod_hierarchical_moe.pt`: 사용자가 학습 후 추가할 체크포인트

## 로컬 실행

Python 3.10 이상을 권장합니다.

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

macOS/Linux:

```bash
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

브라우저에서 보통 `http://localhost:8501`이 열립니다.

## 체크포인트 사용

학습 노트북에서 생성된 파일:

```text
tie_rod_hierarchical_moe.pt
```

을 다음 중 하나의 방법으로 사용합니다.

1. `app.py`와 같은 폴더에 복사
2. 웹 화면 왼쪽 사이드바에서 직접 업로드

> `.pt`는 임의 코드 실행 위험이 있을 수 있으므로 본인이 직접 학습한 파일만 사용하세요.

## 주요 기능

- 네 가지 타이로드 형상 선택
- 형상에 맞는 입력창 자동 표시
- 좌굴하중 N/kN 예측
- 학습범위 외삽 경고
- 공유·형상별 전문가 게이트 가중치 표시
- CSV 일괄 예측 및 결과 다운로드

## 현재 모델의 한계

- Straight-Fuse만 실제 FEA 결과를 사용했습니다.
- Curved-Fuse, Offset-Fuse, Curved-Non Fuse는 가상 데이터 기반입니다.
- 실제 설계 확정 전에는 반드시 FEA 또는 시험으로 재검증해야 합니다.
