# 필요한 라이브러리를 불러옵니다.
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

# 한국어 폰트 설정을 위한 코드 (Matplotlib에서 한국어 깨짐 방지)
# 실행 환경에 따라 폰트가 없을 경우, 다른 폰트로 변경하거나 폰트 설치가 필요할 수 있습니다.
try:
    plt.rc('font', family='Malgun Gothic')
    # Retina 디스플레이의 경우 글씨가 흐릿하게 보이는 현상 방지
    plt.rc('figure', dpi=100)
except:
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
print(f"모델의 기울기 (w): {slope:.2f}")
print(f"모델의 y절편 (b): {intercept:.2f}")
print(f"-> 회귀식: y = {slope:.2f} * X + {intercept:.2f}")


# 4. 모델을 사용한 예측
# 학습에 사용된 X값 전체에 대한 예측값을 구합니다.
y_pred = model.predict(X)

# 새로운 데이터(예: 9시간 공부)에 대한 점수를 예측합니다.
new_X = np.array([[9]])
predicted_score = model.predict(new_X)
print(f"\n9시간 공부했을 때의 예상 시험 점수: {predicted_score[0]:.2f}점")


# 5. 모델 평가
# 결정 계수(R-squared)를 계산하여 모델의 설명력을 평가합니다.
# 1에 가까울수록 모델이 데이터를 잘 설명한다는 의미입니다.
r2 = r2_score(y, y_pred)
print(f"결정 계수 (R-squared): {r2:.2f}")


# 6. 결과 시각화
plt.figure(figsize=(10, 6))

# 실제 데이터 (파란색 점)
plt.scatter(X, y, color='blue', label='실제 데이터')

# 모델이 예측한 회귀선 (빨간색 선)
plt.plot(X, y_pred, color='red', linewidth=2, label='회귀선')

# 9시간 공부했을 때의 예측값 (녹색 점)
plt.scatter(new_X, predicted_score, color='green', s=150, zorder=5, label='9시간 공부 시 예측 점수')

# 그래프 제목 및 라벨 설정
plt.title('공부 시간에 따른 시험 점수 회귀분석', fontsize=16)
plt.xlabel('공부 시간 (X)', fontsize=12)
plt.ylabel('시험 점수 (y)', fontsize=12)
plt.legend()
plt.grid(True)
plt.show()
