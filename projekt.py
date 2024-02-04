import pandas as pd

df = pd.read_csv('df_to_dash.csv')


import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import plotly.express as px
import plotly.figure_factory as ff

from sklearn.preprocessing import StandardScaler

one_hot_encoded = pd.get_dummies(df, dtype=int)
scaler = StandardScaler()
scaler.fit(one_hot_encoded)

one_hot_encoded = pd.DataFrame(scaler.transform(one_hot_encoded),columns=one_hot_encoded.columns.tolist())


st.set_page_config(layout = "wide")

page = st.sidebar.selectbox('Select page',['Modele','Wizualizacja']) 

if page == 'Modele':
    
    st.header("Dane")
    st.dataframe(df)
    
    st.header("Dane encoded i scaled")
    st.dataframe(one_hot_encoded)

    st.header("Model")

    clist = one_hot_encoded.columns.tolist()

    var = st.selectbox("Wybierz zmianną od jakiej ma być zależna cena:",clist)

    y = one_hot_encoded[' price']
    X = sm.add_constant(one_hot_encoded[f'{var}'])

    lm = sm.OLS(y, X)
    lm_fit = lm.fit()
    st.write(lm_fit.summary())
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    one_hot_encoded.plot(kind='scatter', y=' price', x=f'{var}', ax=ax1)
    (b0, b1) = lm_fit.params

    ax1.axline(xy1=(0,b0), slope=b1, color='k')

    ax2.scatter(one_hot_encoded[' price'], lm_fit.resid)
    ax2.axhline(0, linestyle='--', color='k')
    ax2.set_xlabel(' price')
    ax2.set_ylabel('Residual')
    st.pyplot(plt.gcf())
    
else:
    st.header("Zależności")
    

    fig_corr = px.imshow(one_hot_encoded.corr(), labels=dict(color="Korelacja"), x=one_hot_encoded.columns,
                         y=one_hot_encoded.columns,text_auto=True )
    fig_corr.update_layout(
    width=1100,
    height=1100,
    )
    st.plotly_chart(fig_corr)
    
    clist = df.columns.tolist()
    var = st.selectbox("Wybierz zmianną od wizualizacji",clist)
    
    st.header("Liczebność kategorii")
    fig_hist = px.histogram(df, x=f'{var}', labels={'x': f'{var}', 'y': 'Liczność'})
    st.plotly_chart(fig_hist)
    
    if df[f'{var}'].dtype == 'float64' or df[f'{var}'].dtype == 'int':
        st.header("Rozkład zmiennej")
        fig, ax = plt.subplots(figsize=(12, 12))
        sns.distplot(df[[f'{var}']])
        st.pyplot(fig)
    
    st.header("Box plot")
    fig_box = px.box(df, x=var, labels={'x': var})
    st.plotly_chart(fig_box)
    
    st.header("Scatter plot")
    fig_sc = px.scatter(df, x=var)
    st.plotly_chart(fig_sc)