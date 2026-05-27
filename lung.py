import matplotlib.pyplot as plt
import platform

# 폰트 설정: Windows와 Mac 환경에 맞춰 자동 설정
if platform.system() == 'Windows':
    plt.rc('font', family='Malgun Gothic')
elif platform.system() == 'Darwin': # Mac
    plt.rc('font', family='AppleGothic')
else:
    plt.rc('font', family='NanumGothic')

# 마이너스 기호(-) 깨짐 방지
plt.rcParams['axes.unicode_minus'] = False
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 1. 페이지 설정 및 디자인 (깔롱 쌈뽕한 CSS)
st.set_page_config(page_title="Lung Health AI Deluxe", page_icon="🫁", layout="centered")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;400;700&display=swap');
    * { font-family: 'Noto Sans KR', sans-serif; }
    
    .main {
        background: linear-gradient(180deg, #f0f2f6 0%, #dfe9f3 100%);
    }
    
    /* 버튼 스타일 */
    .stButton>button {
        width: 100%;
        border-radius: 15px;
        height: 3.5em;
        background: linear-gradient(45deg, #00c6ff, #0072ff);
        color: white;
        font-weight: bold;
        border: none;
        box-shadow: 0 4px 15px rgba(0, 114, 255, 0.3);
        transition: 0.3s;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0, 114, 255, 0.4);
    }

    /* 결과 카드 스타일 */
    .result-card {
        padding: 25px;
        border-radius: 20px;
        background: white;
        box-shadow: 0 10px 25px rgba(0,0,0,0.05);
        text-align: center;
        border: 1px solid #e0e0e0;
    }
    </style>
""", unsafe_allow_html=True)

# 2. 모델 및 데이터 로드 함수
@st.cache_resource
def load_assets():
    try:
        # 파일명 확인: lung_cancer_model.pkl, lung_cancer_scaler.pkl, lung.csv
        m = joblib.load("lung_cancer_model.pkl")
        s = joblib.load("lung_cancer_scaler.pkl")
        d = pd.read_csv("lung.csv") 
        return m, s, d
    except Exception as e:
        return None, None, None

model, scaler, df = load_assets()

# 3. 헤더 섹션
st.write("### 🛡️ AI 프리미엄 헬스케어")
st.title("폐암 위험군 분류 분석 리포트")
st.markdown("사용자의 생활 습관 데이터를 기반으로 **AI가 군집 분석**을 수행합니다.")
st.divider()

# 4. 입력 레이아웃
if df is not None:
    with st.container():
        st.subheader("📍 분석 데이터 입력")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            smokes = st.number_input("🚬 흡연량 (개비/일)", 0, 50, 10)
        with col2:
            areaq = st.slider("🏡 지역 오염도", 1, 10, 5)
        with col3:
            alkhol = st.number_input("🍺 음주량 (잔/주)", 0, 50, 5)

    # 5. 분석 실행
    st.write("")
    if st.button("실시간 AI 분석 및 위치 파악 시작"):
        if model and scaler:
            # 예측 수행
            # (학습 때 사용한 컬럼명 '흡연', '지역점수', '음주'와 일치해야 함)
            input_df = pd.DataFrame([[smokes, areaq, alkhol]], columns=['흡연', '지역점수', '음주'])
            scaled = scaler.transform(input_df)
            res = model.predict(scaled)[0]
            
            # 결과 지표 출력
            st.subheader("📊 분석 결과 요약")
            m1, m2, m3 = st.columns(3)
            m1.metric("흡연 수치", f"{smokes}")
            m2.metric("환경 점수", f"{areaq}/10")
            m3.metric("음주 수치", f"{alkhol}")

            # 결과 메시지
            st.markdown("<div class='result-card'>", unsafe_allow_html=True)
            if res == 1:
                st.error(f"### 🚨 경고: 군집 {res} (고위험군)")
                st.write("정밀 검진이 필요한 상태입니다. 전문의와 상담을 권장합니다.")
            elif res == 0:
                st.warning(f"### ⚠️ 주의: 군집 {res} (중간위험군)")
                st.write("현재 생활 습관의 개선이 필요합니다. 금연과 절주를 권장합니다.")
            else:
                st.success(f"### ✅ 안전: 군집 {res} (저위험군)")
                st.write("매우 양호한 상태입니다! 현재의 건강한 습관을 유지하세요.")
            st.markdown("</div>", unsafe_allow_html=True)

            # 6. 시각화 (Matplotlib + Seaborn)
            st.divider()
            st.subheader("📍 군집 내 나의 위치 확인")
            
            fig, ax = plt.subplots(figsize=(10, 6))
            # 한글 폰트 설정 (에러 방지를 위해 영문 권장하나, 필요시 한글 설정 추가)
            plt.rcParams['font.family'] = 'Malgun Gothic' # Windows 기준
            
            # 배경 데이터 산점도 (기존 lung.csv 데이터)
            # csv 파일에 '흡연', '지역점수', 'cluster' 컬럼이 있어야 합니다.
            sns.scatterplot(data=df, x='흡연', y='지역점수', hue='cluster', 
                            palette='viridis', alpha=0.5, ax=ax)
            
            # 사용자 위치 표시 (빨간색 큰 X)
            ax.scatter(smokes, areaq, color='red', s=500, marker='X', 
                       edgecolor='black', label='My Position', zorder=5)
            
            ax.set_title("폐암 위험군 군집 분포도", fontsize=15)
            ax.set_xlabel("흡연량", fontsize=12)
            ax.set_ylabel("지역 오염도", fontsize=12)
            ax.legend()
            
            st.pyplot(fig)
            
        else:
            st.error("❌ 모델 또는 스케일러 파일을 불러올 수 없습니다.")
else:
    st.error("❌ `lung.csv` 파일을 찾을 수 없습니다. 파일명을 확인해 주세요.")

# 7. 푸터
st.markdown("""
    <br><br>
    <p style='text-align: center; color: gray; font-size: 0.8em;'>
    © 2024 AI Healthcare Dashboard | 쌈뽕한 분석 템플릿
    </p>
""", unsafe_allow_html=True)
