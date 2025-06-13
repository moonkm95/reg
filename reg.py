# 필요한 라이브러리를 불러옵니다.
import streamlit as st
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import matplotlib.font_manager as fm
import os

# --- 한글 폰트 설정 (Streamlit Cloud) ---
@st.cache_data
def font_setup():
    # Streamlit Cloud 서버에 나눔고딕 폰트 설치 확인 및 경로 설정
    # packages.txt를 통해 fonts-nanum*이 먼저 설치되어야 합니다.
    font_path = '/usr/share/fonts/truetype/nanum/NanumGothic.ttf'
    if os.path.exists(font_path):
        fm.fontManager.addfont(font_path)
        plt.rc('font', family='NanumGothic')
    else:
        # 로컬 환경 (예: Windows) 또는 폰트 경로가 다를 경우를 위한 대체 설정
        try:
            plt.rc('font', family='Malgun Gothic')
        except:
            # 기본 폰트로 fallback
            pass
    # 마이너스 기호 깨짐 방지 설정
    plt.rc('axes', unicode_minus=False)

font_setup()
# -----------------------------------------

# --- Streamlit 앱 설정 ---
st.title('공부 시간에 따른 시험 점수 회귀분석')
st.write("간단한 선형 회귀 모델을 만들어 공부 시간에 따른 시험 점수를 예측하는 웹 앱입니다.")

# 1. 예제 데이터 생성
X = np.array([2, 4, 5, 7, 8, 10, 11, 12]).reshape(-1, 1)
y = np.array([50, 65, 70, 80, 88, 92, 95, 98])


# 2. 선형 회귀 모델 생성 및 학습
model = LinearRegression()
model.fit(X, y)


# 3. 모델의 회귀 계수(기울기) 및 절편 확인
slope = model.coef_[0]
intercept = model.intercept_

st.subheader("학습된 회귀 모델 정보")
col1, col2, col3 = st.columns(3)
col1.metric("기울기 (w)", f"{slope:.2f}")
col2.metric("y절편 (b)", f"{intercept:.2f}")

y_pred = model.predict(X)
r2 = r2_score(y, y_pred)
col3.metric("결정 계수 (R²)", f"{r2:.2f}")

st.write(f"**회귀식:** `y = {slope:.2f} * X + {intercept:.2f}`")


# 4. 모델을 사용한 예측
st.subheader("새로운 데이터로 점수 예측하기")
new_study_time = st.slider("예측할 공부 시간을 선택하세요:", min_value=1, max_value=15, value=9, key="study_time_slider")
new_X = np.array([[new_study_time]])
predicted_score = model.predict(new_X)
st.success(f"**{new_study_time}시간** 공부했을 때의 예상 시험 점수는 **{predicted_score[0]:.2f}점** 입니다.")


# 6. 결과 시각화
st.subheader("회귀분석 결과 시각화")
fig, ax = plt.subplots(figsize=(10, 6))

ax.scatter(X, y, color='blue', label='실제 데이터')
ax.plot(X, y_pred, color='red', linewidth=2, label='회귀선')
ax.scatter(new_X, predicted_score, color='green', s=150, zorder=5, label=f'{new_study_time}시간 공부 시 예측 점수')

ax.set_title('공부 시간에 따른 시험 점수 회귀분석', fontsize=16)
ax.set_xlabel('공부 시간 (X)', fontsize=12)
ax.set_ylabel('시험 점수 (y)', fontsize=12)
ax.legend()
ax.grid(True)

st.pyplot(fig)
