# 导入需要的库
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Streamlit 用户界面
st.title("Risk Prediction of Early Keratoconus")
# 创建输入框
st.subheader("Please enter the following eigenvalues:")
a = st.number_input("D-index:", min_value=-10.0, max_value=100.0,step=0.1)
b = st.number_input("BE:", min_value=-50.0, max_value=200.0, step=0.1)
c = st.number_input("Kmax(D):", min_value=0.0, max_value=100.0, step=0.1)
d = st.number_input("Galectin-1(pg/mL):", min_value=0.0, max_value=50000.0, step=0.1)
e = st.number_input("Galectin-3(pg/mL):", min_value=0.0, max_value=50000.0, step=0.1)
f = st.number_input("IL-1 beta(pg/mL):", min_value=0.0, max_value=50000.0,step=0.1)

# 如果按下按钮
if st.button("Predict"):  # 显示按钮
    # 加载训练好的模型
    model = joblib.load("random_forest_model.pkl")
    # 将输入存储DataFrame
    X = pd.DataFrame([[a,b,c,d,e,f]],
                     columns = ['d_index','be','kmax','galectin_1','galectin_3','il_1_beta'])
    # 进行预测
    Predict_proba = model.predict_proba(X)[:, 2][0] # 输出2的概率
    st.subheader(f"early_keratoconus_risk:  {'%.2f' % float(Predict_proba * 100) + '%'}")

   # 绘制概率图
    not_probability = 1 - Predict_proba
    # 数据
    sizes = [Predict_proba, not_probability]  # 饼状图的大小
    labels = ['early_keratoconus_risk', ' ']  # 标签
    colors = ['#66c2a5', '#fc8d62']  # 自定义颜色
    explode = (0.1, 0)  # 突出显示Predict_proba部分

    # 创建饼状图
    plt.figure(figsize=(3, 3),dpi=300)  # 设置图形大小
    plt.pie(sizes, explode=explode, labels=labels, colors=colors,
            autopct='%1.2f%%', shadow=True, startangle=90)
    plt.axis('equal')  # 使饼图为圆形
    st.pyplot(plt)
