import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
import os

# 페이지 설정
st.set_page_config(page_title="Lung Health AI Deluxe", page_icon="🫁", layout="centered")

# 커스텀 CSS (세련된 디자인)
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stButton>button {
        width: 100%; border-radius: 12px; height: 3.5em;
        background: linear-gradient(45deg, #00c6ff, #0072ff);
        color: white; font-weight: bold; border: none;
        box-shadow: 0 4px 15px rgba(0, 114, 255, 0.3);
    }
    .result-card {
        padding: 25px; border-radius: 20px; background: white;
        box-shadow: 0 10px 25px rgba(0,0,0,0.05); text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# 모델 및 데이터 로드 (에러 방지 로직 강화)
@st.cache_resource
def load_assets():
    try:
        # 파일 존재 여부 확인 후 로드
        m_path = "lung_cancer_model.pkl"
        s_path = "lung_cancer_scaler.pkl"
        d_path = "lung.csv"
        
        if all(os.path.exists(p) for p in [m_path, s_path, d_path]):
            m = joblib.load(m_path)
            s = joblib.load(s_path)
            d = pd.read_csv(d_path)
            return m, s, d
        else:
            return None, None, None
    except:
        return None, None, None

model, scaler, df = load_assets()

# 헤더
st.write("### 🛡️ AI Premium Healthcare")
st.title("폐암 위험군 분류 분석 리포트")
st.info("데이터를 입력하면 AI가 실시간으로 건강 상태를 분석합니다.")

if df is not None:
    # 입력 섹션
    with st.container():
        st.subheader("📍 환자 데이터 입력")
        c1, c2, c3 = st.columns(3)
        with c1: smokes = st.number_input("🚬 흡연량 (개비)", 0, 50, 10)
        with c2: areaq = st.slider("🏡 지역 오염도", 1, 10, 5)
        with c3: alkhol = st.number_input("🍺 음주량 (잔)", 0, 50, 5)

    if st.button("실시간 AI 분석 시작"):
        # 분석 수행
        # 주의: 컬럼명이 학습 데이터와 정확히 일치해야 함 (흡연, 지역점수, 음주)
        input_data = pd.DataFrame([[smokes, areaq, alkhol]], columns=['흡연', '지역점수', '음주'])
        scaled_data = scaler.transform(input_data)
        res = model.predict(scaled_data)[0]

        # 결과 리포트
        st.divider()
        st.subheader("📊 분석 결과")
        
        st.markdown("<div class='result-card'>", unsafe_allow_html=True)
        if res == 1:
            st.error(f"### 🚨 고위험군 (Cluster {res})")
            st.write("위험도가 높습니다. 빠른 시일 내에 전문의 검진을 권장합니다.")
        elif res == 0:
            st.warning(f"### ⚠️ 주의군 (Cluster {res})")
            st.write("생활 습관 개선이 필요합니다. 금연과 운동을 시작하세요.")
        else:
            st.success(f"### ✅ 안전군 (Cluster {res})")
            st.write("매우 건강한 상태입니다. 현재 습관을 유지하세요!")
        st.markdown("</div>", unsafe_allow_html=True)

        # Plotly 시각화 (한글 깨짐 없음!)
        st.subheader("📍 군집 내 위치 분석")
        
        # 1. 배경 산점도
        fig = px.scatter(df, x='흡연', y='지역점수', color='cluster',
                         color_continuous_scale='Viridis',
                         opacity=0.5, title="전체 데이터 대비 내 위치")
        
        # 2. 내 위치 추가 (빨간색 큰 별점)
        fig.add_trace(go.Scatter(
            x=[smokes], y=[areaq],
            mode='markers',
            marker=dict(color='red', size=18, symbol='x', line=dict(width=2, color='white')),
            name='내 위치'
        ))

        fig.update_layout(
            xaxis_title="흡연량 (개비)",
            yaxis_title="지역 오염도",
            legend_title="위험군",
            template="plotly_white"
        )
        
        st.plotly_chart(fig, use_container_width=True)

else:
    st.error("❗ 필수 파일(lung.csv, pkl 파일들)이 없습니다. GitHub에 파일을 모두 올렸는지 확인해주세요.")

# 푸터
st.markdown("<p style='text-align: center; color: gray; font-size: 0.8em; margin-top: 50px;'>© 2024 AI Healthcare System</p>", unsafe_allow_html=True)
