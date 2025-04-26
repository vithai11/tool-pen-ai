import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBClassifier

st.title("Dự đoán Penalty AI tự học (bản mạnh)")

# Data ban đầu
if 'data' not in st.session_state:
    st.session_state.data = pd.DataFrame(columns=['player_shot', 'gk_move'])

# Model ban đầu
if 'model' not in st.session_state:
    st.session_state.model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
    st.session_state.model.fit([[0, 0], [1, 1]], [0, 1])  # Fit fake đúng 2 class

# Map
move_mapping = {'Trái': 0, 'Giữa': 1, 'Phải': 2}
reverse_mapping = {0: 'Trái', 1: 'Giữa', 2: 'Phải'}

# Người chơi chọn
st.subheader("1. Bạn chọn hướng sút:")
col1, col2, col3 = st.columns(3)
with col1:
    player_shot = st.button('⬅️ Trái')
with col2:
    player_shot_center = st.button('⬆️ Giữa')
with col3:
    player_shot_right = st.button('➡️ Phải')

# Thủ môn
st.subheader("2. Thủ môn nhảy hướng nào?")
col4, col5, col6 = st.columns(3)
with col4:
    gk_left = st.button('🖐️ Trái')
with col5:
    gk_center = st.button('🖐️ Giữa')
with col6:
    gk_right = st.button('🖐️ Phải')

# Lưu kết quả
if player_shot or player_shot_center or player_shot_right:
    player_move = 0 if player_shot else 1 if player_shot_center else 2

    if gk_left:
        gk_move = 0
    elif gk_center:
        gk_move = 1
    elif gk_right:
        gk_move = 2
    else:
        gk_move = None

    if gk_move is not None:
        new_data = pd.DataFrame([[player_move, gk_move]], columns=['player_shot', 'gk_move'])
        st.session_state.data = pd.concat([st.session_state.data, new_data], ignore_index=True)

        # Train lại model nếu có nhiều hơn 5 dữ liệu
        if len(st.session_state.data) >= 5:
            X = st.session_state.data[['player_shot']]
            y = st.session_state.data['gk_move']
            st.session_state.model.fit(X, y)

        # Dự đoán hướng thủ môn
        pred = st.session_state.model.predict([[player_move]])
        result = "Đúng dự đoán" if pred[0] != gk_move else "Sai dự đoán"
        st.write(f"Dự đoán của AI: {result}")

# Gợi ý cho lần kế tiếp
if len(st.session_state.data) >= 5:
    next_moves = []
    for i in range(3):
        proba = st.session_state.model.predict_proba([[i]])[0]
        success_rate = 1 - proba[i]
        next_moves.append((i, success_rate))

    best_move = max(next_moves, key=lambda x: x[1])

    st.subheader("Gợi ý từ AI (cho lượt kế tiếp):")
    st.success(f"Nên sút về: {reverse_mapping[best_move[0]]}")
    st.write(f"Tỷ lệ sút thành công: {best_move[1]*100:.2f}%")