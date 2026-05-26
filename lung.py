import streamlit as st
import pandas as pd
import joblib
import os

# 페이지 설정
st.set_page_config(page_title="Lung Health AI", page_icon="🚀", layout="centered")

# 커스텀 CSS (깔롱한 디자인 입히기)
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        width: 100%;
        border-radius: 20px;
        height: 3em;
        background-color: #ff4b4b;
        color: white;
        font-weight: bold;
        border: none;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #ff2b2b;
        transform: scale(1.02);
    }
    .result-card {
        padding: 20px;
        border-radius: 15px;
        background-color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# 모델 로드
@st.cache_resource
def load_assets():
    try:
        m = joblib.load("lung_cancer_model.pkl")
        s = joblib.load("lung_cancer_scaler.pkl")
        return m, s
    except:
        return None, None

model, scaler = load_assets()

# 헤더 섹션
st.write("### 🛡️ AI 헬스케어 시스템")
st.title("폐암 위험군 분류 분석")
st.info("입력 데이터를 분석하여 인공지능이 건강 군집을 예측합니다.")

# 입력 레이아웃
with st.container():
    st.subheader("📍 환자 데이터 입력")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        smokes = st.number_input("🚬 흡연량", 0, 50, 10)
    with col2:
        areaq = st.slider("🏡 지역오염", 1, 10, 5)
    with col3:
        alkhol = st.number_input("🍺 음주량", 0, 50, 5)

# 예측 버튼
st.markdown("--- ")
if st.button("실시간 AI 분석 시작"):
    if model and scaler:
        # 예측 수행
        input_df = pd.DataFrame([[smokes, areaq, alkhol]], columns=['흡연', '지역점수', '음주'])
        scaled = scaler.transform(input_df)
        res = model.predict(scaled)[0]
        
        # 결과 시각화
        st.subheader("📊 분석 리포트")
        m1, m2, m3 = st.columns(3)
        m1.metric("흡연 수치", f"{smokes} 개비")
        m2.metric("지역 등급", f"{areaq}/10")
        m3.metric("음주 수치", f"{alkhol} 잔")

        st.markdown(f"<div class='result-card'>", unsafe_allow_html=True)
        if res == 1:
            st.error(f"### 🚨 경고: 군집 {res} (고위험군)")
            st.write("현재 데이터는 건강 위험도가 높은 그룹에 가깝습니다. 정밀 진단을 권장합니다.")
        elif res == 0:
            st.warning(f"### ⚠️ 주의: 군집 {res} (중간위험군)")
            st.write("생활 습관 교정이 필요한 그룹입니다. 꾸준한 관리가 필요합니다.")
        else:
            st.success(f"### ✅ 안전: 군집 {res} (저위험군)")
            st.write("비교적 건강한 그룹에 속해 있습니다. 현재 상태를 유지하세요!")
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.error("모델 파일(.pkl)을 찾을 수 없습니다.")

# 푸터
st.markdown("""
    <br><br>
    <p style='text-align: center; color: gray; font-size: 0.8em;'>
    © 2024 AI Medical Dashboard | VS Code Streamlit Template
    </p>
""", unsafe_allow_html=True)