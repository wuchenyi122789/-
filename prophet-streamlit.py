import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import io
import holidays

# 设置页面配置
st.set_page_config(layout="wide", page_title="优化后的云厨房销售预测系统")

# 节假日数据
jp_holidays = holidays.JP()

# 预测销售的主函数
@st.cache_data(ttl="1h", show_spinner=False)
def predict_sales(order_dates, daily_net_income, changepoint_sensitivity, seasonality_sensitivity, 
                  promo_strengths, promo_durations, offset, may_adjustment, cap, floor, forecast_months,
                  weather_impact, holiday_impact, weekday_impact, optimize_future=False):
    
    # 准备数据
    dates = pd.to_datetime(order_dates)
    df = pd.DataFrame({'ds': dates, 'y': daily_net_income})

    # 添加容量（cap）和下限（floor）
    df['cap'] = cap
    df['floor'] = floor

    # 调整5月销量
    df.loc[df['ds'].dt.month == 5, 'y'] *= may_adjustment

    # 添加节假日信息
    df['holiday'] = df['ds'].apply(lambda x: jp_holidays.get(x, 0))
    df['is_holiday'] = df['holiday'].astype(bool).astype(int)

    # 添加天气影响（这里使用随机数模拟，实际应用中应该使用真实天气数据）
    np.random.seed(42)
    df['temp'] = np.random.normal(20, 5, size=len(df))
    df['rain'] = np.random.choice([0, 1], size=len(df), p=[0.8, 0.2])

    # 设置促销日期
    promo_dates = ['2024-06-09', '2024-06-21', '2024-07-13']
    holidays = pd.DataFrame({
        'holiday': ['promo1', 'promo2', 'promo3'],
        'ds': pd.to_datetime(promo_dates),
        'lower_window': [0, 0, 0],
        'upper_window': promo_durations,
        'prior_scale': promo_strengths
    })

    # 创建和配置模型
    model = Prophet(
        changepoint_prior_scale=changepoint_sensitivity,
        seasonality_prior_scale=seasonality_sensitivity,
        weekly_seasonality=True,
        daily_seasonality=True,
        holidays=holidays,
        interval_width=0.95,
        growth='logistic'  # 使用逻辑增长
    )
    
    # 添加额外的季节性组件
    model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
    model.add_seasonality(name='quarterly', period=91.25, fourier_order=5)
    
    # 添加回归变量
    model.add_regressor('temp', mode='multiplicative')
    model.add_regressor('rain', mode='multiplicative')
    model.add_regressor('is_holiday', mode='multiplicative')
    
    # 添加星期几作为回归变量
    df['weekday'] = df['ds'].dt.weekday
    model.add_regressor('weekday', mode='multiplicative')

    # 拟合模型
    model.fit(df)

    # 创建未来数据框
    future_periods = forecast_months * 30
    future = model.make_future_dataframe(periods=future_periods, include_history=True)
    future['cap'] = cap
    future['floor'] = floor
    
    # 为未来数据添加特征
    future['weekday'] = future['ds'].dt.weekday
    future['holiday'] = future['ds'].apply(lambda x: jp_holidays.get(x, 0))
    future['is_holiday'] = future['holiday'].astype(bool).astype(int)
    future['temp'] = np.random.normal(20, 5, size=len(future))
    future['rain'] = np.random.choice([0, 1], size=len(future), p=[0.8, 0.2])

    # 如果需要优化未来促销强度
    if optimize_future:
        def objective_function(promo_strength):
            holidays_future = pd.DataFrame({
                'holiday': ['future_promo'] * future_periods,
                'ds': future['ds'][-future_periods:],
                'lower_window': [0] * future_periods,
                'upper_window': [0] * future_periods,
                'prior_scale': [promo_strength[0]] * future_periods
            })
            model.holidays = pd.concat([holidays, holidays_future])
            forecast = model.predict(future)
            return -forecast['yhat'][-future_periods:].mean()

        result = minimize(objective_function, x0=[0.5], bounds=[(0.0, 2.0)], method='L-BFGS-B')
        best_promo_strength = result.x[0]

        holidays_future_optimal = pd.DataFrame({
            'holiday': ['future_promo'] * future_periods,
            'ds': future['ds'][-future_periods:],
            'lower_window': [0] * future_periods,
            'upper_window': [0] * future_periods,
            'prior_scale': [best_promo_strength] * future_periods
        })
        model.holidays = pd.concat([holidays, holidays_future_optimal])
    else:
        best_promo_strength = None

    # 生成预测
    forecast = model.predict(future)
    forecast['yhat'] += offset

    # 应用天气、节假日和星期几的影响
    forecast['yhat'] *= (1 + weather_impact * (forecast['temp'] - 20) / 20)
    forecast['yhat'] *= (1 + weather_impact * forecast['rain'])
    forecast['yhat'] *= (1 + holiday_impact * forecast['is_holiday'])
    for i in range(7):
        forecast.loc[forecast['weekday'] == i, 'yhat'] *= (1 + weekday_impact[i])

    return forecast, model, best_promo_strength

# 聚类分析函数
def cluster_analysis(df):
    features = ['yhat', 'weekday', 'is_holiday', 'temp', 'rain']
    X = df[features]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    kmeans = KMeans(n_clusters=3, random_state=42)
    df['cluster'] = kmeans.fit_predict(X_scaled)
    
    return df

# Streamlit 用户界面
st.title('云厨房收入预测系统')

# 上传文件
uploaded_file = st.file_uploader("上传您的数据文件", type=["xlsx", "csv"])
if uploaded_file:
    df = pd.read_excel(uploaded_file) if uploaded_file.name.endswith('xlsx') else pd.read_csv(uploaded_file)
    df['订单日期'] = pd.to_datetime(df['订单日期'])

    # 创建两列布局
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("参数调整")
        changepoint_prior_scale = st.slider('趋势变化点灵敏度', 0.001, 0.5, 0.05, step=0.001)
        seasonality_prior_scale = st.slider('季节性灵敏度', 0.01, 10.0, 1.0, step=0.01)
        
        st.subheader("促销参数")
        promo_strengths = [
            st.slider('促销1强度', 0.0, 2.0, 0.5, step=0.05),
            st.slider('促销2强度', 0.0, 2.0, 0.5, step=0.05),
            st.slider('促销3强度', 0.0, 2.0, 0.5, step=0.05)
        ]
        promo_durations = [
            st.slider('促销1影响天数', 1, 60, 7),
            st.slider('促销2影响天数', 1, 60, 7),
            st.slider('促销3影响天数', 1, 60, 7)
        ]
        
        st.subheader("其他参数")
        offset = st.number_input('偏移量', min_value=-10000, max_value=10000, value=0, step=100)
        may_adjustment = st.slider('五月调整系数', 0.5, 2.0, 1.0, step=0.01)
        cap = st.number_input('收入上限（逻辑增长上限）', value=float(df['日净收入'].max() * 2), step=1000.0)
        floor = st.number_input('收入下限（逻辑增长下限）', value=float(df['日净收入'].min() * 0.5), step=100.0)
        forecast_months = st.slider('预测未来月数', 1, 24, 3)
        
        st.subheader("新增参数")
        weather_impact = st.slider('天气影响程度', 0.0, 0.5, 0.1, step=0.01)
        holiday_impact = st.slider('节假日影响程度', 0.0, 0.5, 0.1, step=0.01)
        weekday_impact = [st.slider(f'星期{i}影响', -0.2, 0.2, 0.0, step=0.01) for i in range(7)]
        
        optimize_future = st.checkbox('优化未来促销强度', value=True)

    with col2:
        # 实时预测
        forecast, model, best_promo_strength = predict_sales(
            df['订单日期'], df['日净收入'], 
            changepoint_prior_scale, seasonality_prior_scale, 
            promo_strengths, promo_durations, 
            offset, may_adjustment, cap, floor, forecast_months,
            weather_impact, holiday_impact, weekday_impact, optimize_future
        )

        st.subheader("预测图表")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='预测值', line=dict(color='red')))
        fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], mode='lines', name='预测下限', line=dict(color='rgba(255,0,0,0.2)')))
        fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], mode='lines', name='预测上限', line=dict(color='rgba(255,0,0,0.2)')))
        fig.add_trace(go.Scatter(x=df['订单日期'], y=df['日净收入'], mode='lines', name='历史数据', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=forecast['ds'], y=[cap]*len(forecast), mode='lines', name='增长上限', line=dict(color='green', dash='dash')))
        fig.add_trace(go.Scatter(x=forecast['ds'], y=[floor]*len(forecast), mode='lines', name='增长下限', line=dict(color='orange', dash='dash')))
        fig.update_layout(height=600, width=1000, xaxis_title='日期', yaxis_title='日净收入')
        st.plotly_chart(fig)

        if optimize_future:
            st.subheader(f"优化后的未来促销强度: {best_promo_strength:.4f}")

        # 聚类分析
        st.subheader("聚类分析")
        clustered_df = cluster_analysis(forecast)
        fig_cluster = go.Figure()
        for i in range(3):
            cluster_data = clustered_df[clustered_df['cluster'] == i]
            fig_cluster.add_trace(go.Scatter(x=cluster_data['ds'], y=cluster_data['yhat'], mode='markers', name=f'聚类 {i}'))
        fig_cluster.update_layout(height=400, width=1000, xaxis_title='日期', yaxis_title='预测销售额')
        st.plotly_chart(fig_cluster)

    # 交互式预测探索
    st.subheader("交互式预测探索")
    selected_date = st.date_input("选择日期进行预测探索", min_value=forecast['ds'].min(), max_value=forecast['ds'].max())
    selected_forecast = forecast[forecast['ds'] == pd.to_datetime(selected_date)].iloc[0]

    st.write(f"选定日期的预测值: {selected_forecast['yhat']:.2f}")
    st.write(f"预测区间: [{selected_forecast['yhat_lower']:.2f}, {selected_forecast['yhat_upper']:.2f}]")

    # 动态显示组件
    components = ['trend', 'weekly', 'monthly', 'quarterly']
    for component in components:
        if component in selected_forecast:
            st.write(f"{component.capitalize()}: {selected_forecast[component]:.2f}")
        else:
            st.write(f"{component.capitalize()}: 不适用")

    # 显示促销影响
    for i in range(1, 4):
        if f'promo{i}' in selected_forecast:
            st.write(f"促销{i}影响: {selected_forecast[f'promo{i}']:.2f}")

    # 显示其他相关信息
    additional_info = {
        'temp': ('温度', '{:.1f}'),
        'rain': ('是否下雨', lambda x: '是' if x else '否'),
        'is_holiday': ('是否节假日', lambda x: '是' if x else '否'),
        'weekday': ('星期', str)
    }
    for key, (label, formatter) in additional_info.items():
        if key in selected_forecast:
            value = formatter(selected_forecast[key]) if callable(formatter) else formatter.format(selected_forecast[key])
            st.write(f"{label}: {value}")

    # 导出功能
    st.subheader("导出预测结果")
    
    # 准备详细的导出数据
    export_df = pd.DataFrame({
        "日期": forecast['ds'],
        "预测值": forecast['yhat'],
        "预测下限": forecast['yhat_lower'],
        "预测上限": forecast['yhat_upper'],
        "实际值": list(df['日净收入']) + [None] * (len(forecast) - len(df)),
        "趋势": forecast['trend'],
        "每周季节性": forecast['weekly'],
        "每月季节性": forecast['monthly'],
        "每季度季节性": forecast['quarterly'],
        "促销1影响": forecast['promo1'] if 'promo1' in forecast.columns else None,
        "促销2影响": forecast['promo2'] if 'promo2' in forecast.columns else None,
        "促销3影响": forecast['promo3'] if 'promo3' in forecast.columns else None,
        "温度": forecast['temp'],
        "是否下雨": forecast['rain'],
        "是否节假日": forecast['is_holiday'],
        "星期几": forecast['weekday'],
        "聚类": clustered_df['cluster']
    })

    # 添加参数信息
    params_df = pd.DataFrame({
        "参数": ["趋势变化点灵敏度", "季节性灵敏度", "促销1强度", "促销2强度", "促销3强度",
                 "促销1影响天数", "促销2影响天数", "促销3影响天数", "偏移量", "五月调整系数", 
                 "收入上限（逻辑增长上限）", "收入下限（逻辑增长下限）", "预测未来月数", 
                 "天气影响程度", "节假日影响程度"] + [f"星期{i}影响" for i in range(7)] + ["优化后的未来促销强度"],
        "值": [changepoint_prior_scale, seasonality_prior_scale] + promo_strengths + promo_durations + 
             [offset, may_adjustment, cap, floor, forecast_months, 
              weather_impact, holiday_impact] + weekday_impact + 
             [best_promo_strength if optimize_future else "未优化"]
    })

    # 创建一个 BytesIO 对象来保存 Excel 文件
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        export_df.to_excel(writer, sheet_name='预测结果', index=False)
        params_df.to_excel(writer, sheet_name='参数', index=False)

    # 提供下载按钮
    output.seek(0)
    st.download_button(
        label="下载详细预测报告",
        data=output,
        file_name="forecast_detailed_report.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    # 模型性能评估
    st.subheader("模型性能评估")
    df_cv = cross_validation(model, initial='30 days', period='7 days', horizon = '30 days')
    df_p = performance_metrics(df_cv)
    st.write(df_p)

    # 组件贡献分析
    st.subheader("组件贡献分析")
    fig_components = model.plot_components(forecast)
    st.pyplot(fig_components)

# 添加说明信息
st.sidebar.title("使用说明")
st.sidebar.write("""
1. 上传包含'订单日期'和'日净收入'列的Excel或CSV文件。
2. 调整各项参数以优化预测效果。
3. 查看预测图表、聚类分析和模型性能评估。
4. 使用交互式预测探索工具。
5. 下载详细预测报告进行深入分析。
""")

st.sidebar.title("新增功能说明")
st.sidebar.write("""
- 天气影响：模拟天气对销售的影响。
- 节假日影响：考虑节假日对销售的影响。
- 星期影响：每个工作日对销售的不同影响。
- 聚类分析：对预测结果进行聚类，发现潜在模式。
- 交互式预测探索：选择特定日期查看详细预测信息。
- 模型性能评估：通过交叉验证评估模型性能。
- 组件贡献分析：查看各因素对预测的影响。
""")