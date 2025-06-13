# 필요한 라이브러리를 불러옵니다.
import streamlit as st
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

# --- Streamlit 앱 설정 ---
st.title('공부 시간에 따른 시험 점수 회귀분석')
st.write("간단한 선형 회귀 모델을 만들어 공부 시간에 따른 시험 점수를 예측하는 웹 앱입니다.")

# 한국어 폰트 설정을 위한 코드 (Matplotlib에서 한국어 깨짐 방지)
try:
    plt.rc('font', family='Malgun Gothic')
    plt.rc('figure', dpi=100)
except:
    st.warning("'Malgun Gothic' 폰트를 찾을 수 없습니다. 그래프의 한글이 깨질 수 있습니다.")
    pass

# 1. 예제 데이터 생성
# 독립 변수(X): 공부 시간 (시간 단위)
# reshape(-1, 1)은 scikit-learn 모델에 사용하기 위해 2D 배열로 변환하는 과정입니다.
X = np.array([2, 4, 5, 7, 8, 10, 11, 12]).reshape(-1, 1)

# 종속 변수(y): 시험 점수
y = np.array([50, 65, 70, 80, 88, 92, 95, 98])


# 2. 선형 회귀 모델 생성 및 학습
# LinearRegression 모델 객체를 만듭니다.
model = LinearRegression()

# fit() 메서드를 사용하여 모델을 데이터에 학습시킵니다.
model.fit(X, y)


# 3. 모델의 회귀 계수(기울기) 및 절편 확인
# coef_는 기울기(가중치)를, intercept_는 y절편을 나타냅니다.
slope = model.coef_[0]
intercept = model.intercept_

st.subheader("학습된 회귀 모델 정보")
col1, col2, col3 = st.columns(3)
col1.metric("기울기 (w)", f"{slope:.2f}")
col2.metric("y절편 (b)", f"{intercept:.2f}")

# 5. 모델 평가
y_pred = model.predict(X)
r2 = r2_score(y, y_pred)
col3.metric("결정 계수 (R²)", f"{r2:.2f}")

st.write(f"**회귀식:** `y = {slope:.2f} * X + {intercept:.2f}`")


# 4. 모델을 사용한 예측
st.subheader("새로운 데이터로 점수 예측하기")
new_study_time = st.slider("예측할 공부 시간을 선택하세요:", min_value=1, max_value=15, value=9)
new_X = np.array([[new_study_time]])
predicted_score = model.predict(new_X)
st.success(f"**{new_study_time}시간** 공부했을 때의 예상 시험 점수는 **{predicted_score[0]:.2f}점** 입니다.")


# 6. 결과 시각화
st.subheader("회귀분석 결과 시각화")
fig, ax = plt.subplots(figsize=(10, 6))

# 실제 데이터 (파란색 점)
ax.scatter(X, y, color='blue', label='실제 데이터')

# 모델이 예측한 회귀선 (빨간색 선)
ax.plot(X, y_pred, color='red', linewidth=2, label='회귀선')

# 새로운 예측값 (녹색 점)
ax.scatter(new_X, predicted_score, color='green', s=150, zorder=5, label=f'{new_study_time}시간 공부 시 예측 점수')

# 그래프 제목 및 라벨 설정
ax.set_title('공부 시간에 따른 시험 점수 회귀분석', fontsize=16)
ax.set_xlabel('공부 시간 (X)', fontsize=12)
ax.set_ylabel('시험 점수 (y)', fontsize=12)
ax.legend()
ax.grid(True)

# Streamlit에 그래프를 표시합니다.
st.pyplot(fig)
