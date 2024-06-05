import json
import time

import hydralit_components as hc
import pandas as pd
import plotly.express as px
# import snowflake.connector as sf
import streamlit as st
import yfinance as yf
from babel.numbers import format_currency
from openpyxl import Workbook, load_workbook
from streamlit_lottie import st_lottie
from yahooquery import Ticker

from functions import *

def load_lottiefile(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)


lottie_analysis = load_lottiefile("lottiefiles/analysis.json")
# lottie_hello = load_lottiefile("lottiefiles/hello.json")

st.set_page_config(
    page_title="FA",
    page_icon="bar_chart",
    # page_icon="chart_with_upwards_trend",
    layout="wide",
    initial_sidebar_state="expanded"
)

# hide_st_style = """
# <style>
# #MainMenu {visibility: hidden;}
# header{visibility: hidden;}
# footer {visibility: hidden;}
# </style>

# """
# st.markdown(hide_st_style, unsafe_allow_html=True)

# ---Side-Bar----
# original_title = '<p style="font-family:Courier; color:dark-grey; font-size: 20px;">LDR</p>'

with st.sidebar:
    st.image('ldr.png')
    menu = st.selectbox('Pilih Menu',['Analisis Laporan Tahunan','Analisis IPO','Optimasi Portofolio','AI-Financial Assistant'])
    st_lottie(lottie_analysis, height="150",
                  width="250", quality="high", key=None)
    
    st.write("---")
    # st.write("#### About📍")
    # st.write('''
    # This Web App is based on the Beneish model, a mathematical model that uses financial ratios and eight variables to determine whether a company has manipulated earnings. Based on company financial statements, an M-Score is constructed to describe how much earnings have been manipulated.
    # ''')

pd.set_option('mode.chained_assignment', None)
pd.set_option('future.no_silent_downcasting', True)

st.title("Analyzing the Reliability of Company Reports")

# if menu=='Stock Risk Dashboard':
#     pass
# elif menu =='Portfolio Optimization':
#     st.write('test')
if menu =='IPO Analyzer':
    st.write('Dalam Pengembangan')
elif menu =='Optimasi Portofolio':
    st.write('Dalam Pengembangan')
elif menu =='AI-Financial Assistant':
    st.write('Dalam Pengembangan')
elif menu == 'Analisis Laporan Tahunan':
    cekradio = st.sidebar.radio('Pilih',['Upload File','Pilih dari arsip'])
    if cekradio == 'Upload File':
        lc,rc = st.columns((1,1))
        with lc:
            st.subheader('Analisis Data Finansial')
            # st.write('')
            dki = st.expander('Analisis Keuangan dengan Beneish-M Score',expanded=True)
            with dki:
                kode = st.text_input('Masukkan Kode Perusahaan')
                tahun = st.selectbox('Pilih Tahun', ['2023','2022','2021','2020'])
                # sb = st.write('Ambil Data Keuangan')
                if kode!='':
                    bm = getBenData(f'{kode}.JK',int(tahun))
                    dfben=pd.DataFrame({'Index':list(bm[1].keys())[2:-1],'Ratios':list(bm[1].values())[2:-1]})
                    tki,tka=st.columns((3,1))
                    with tki:
                        st.write(bm[0])
                    with tka:
                        st.write(dfben.to_html(index=False), unsafe_allow_html=True)

                    st.write('')
                    st.write('')
                    st.write('')
                    fig = px.bar(
                        dfben,  # Data Frame
                        x="Index",  # Columns from the data frame
                        y="Ratios",
                        title="Financial Ratio Indexes",
                    )
                    st.plotly_chart(fig)
                    if(bm[1]['BMscore'] < -2.22):
                        res = '##### Kemungkinan Manipulasi Keuangan: Rendah'
                        st.write(f"##### M- Score = {round(bm[1]['BMscore'],2)}")
                        st.write(f"{res}")
                        # print(res)
                    elif(bm[1]['BMscore'] < -1.78):
                        res = '##### Kemungkinan Manipulasi Keuangan: Sedang'
                        st.write(f"##### M- Score = {round(bm[1]['BMscore'],2)}")
                        st.write(f"{res}")
                    else:
                        res = " ##### Kemungkinan Manipulasi Keuangan: Tinggi"
                        st.write(f"##### Beneish M- Score = {round(bm[1]['BMscore'],2)}")
                        st.write(f"{res}")
                else:
                    st.write('Silakan isi kode emiten (IDX) untuk menarik data keuangan')
        with rc:
            st.subheader('Analisis Data Teks')
            # st.write('')
            dka = st.expander('Analisis Text Mining dan Prediksi Manipulasi',expanded=True)
            with dka:
                file_uploaded=st.file_uploader('Upload laporan yang akan dianalisis',type='pdf')
                if file_uploaded:
                    texts = readpdf(file_uploaded)
                    data=[]
                    if texts:
                        data.append({"filename":kode,"text":texts})
                        df = pd.DataFrame(data)
                        df1 = assignLm(df)
                        df2 = getwordLm(df)
                        cki,cka = st.columns((3,1))
                        with cki:
                            val = [df1[col][0] for col in df1.iloc[:,3:].columns]
                            name = [col for col in df1.iloc[:,3:].columns]
                            # st.write(val)
                            # st.write(name)
                            color_map = {
                                'negative':"#de425b",'positive':"#488f31",
                                'uncertainty':"#f4bc92",'litigious':"#b6ba74",
                                'strong_modal':"#ec8569",'weak_modal':"#ffedcf",
                                'constraining':"#96ac5a", 'complexity':"#d2ca90"
                            }
                            colors_map = [color_map[category] for category in name]
                            import plotly.graph_objects as go
                            fig = go.Figure(data=[go.Pie(labels=name, values=val,marker_colors=colors_map)])
                            fig.update_layout(title='Proporsi kategori kata')
                            # st.write('Proporsi kategori kata')
                            st.plotly_chart(fig)
                        with cka:
                            st.write('')
                            st.write('')
                            st.write('Jumlah kategori kata')
                            dfone = df1.iloc[:,2:].transpose()
                            dfone.columns = ['Count']
                            st.write(dfone)
                        # st.write(df2.iloc[:,2:].to_html(index=False), unsafe_allow_html=True)
                        cols = df2.columns
                        from ast import literal_eval
                        # dlist = [literal_eval(df2[c]) for c in df2.iloc[:,3:].columns]  # Put the dictionaries into a list
                        dlist = [df2['negative'].values[0],
                                 df2['positive'].values[0],
                                 df2['uncertainty'].values[0],
                                 df2['litigious'].values[0],
                                 df2['strong_modal'].values[0],
                                 df2['weak_modal'].values[0],
                                df2['constraining'].values[0],
                                df2['complexity'].values[0]]
                        # st.write(dlist)
                        mdict = merge_dictionaries(dlist)
                        # st.write('Wordcloud frekuensi kata')
                        wcloud = create_wordcloud(mdict)
                        wcloud.update_layout(title='Wordcloud frekuensi kata')
                        st.plotly_chart(wcloud)
                        df_class = df1[['negative','positive','uncertainty','litigious','strong_modal','weak_modal','constraining', 'complexity']]
                        prediction = get_prediction("model.sav",df_class)
                        pred_score = prediction[1][0][1]
                        if(pred_score < 0.5):
                            res = '##### Kemungkinan Manipulasi Laporan: Rendah'
                            st.write(f"##### Deception Probability = {round(pred_score,2)}")
                            st.write(f"{res}")
                            # print(res)
                        elif(pred_score < 0.75):
                            res = '##### Kemungkinan Manipulasi Laporan: Sedang'
                            st.write(f"##### Deception Probability = {round(pred_score,2)}")
                            st.write(f"{res}")
                        else:
                            res = " ##### Kemungkinan Manipulasi Laporan: Tinggi"
                            st.write(f"##### Deception Probability = {round(pred_score,2)}")
                            st.write(f"{res}")
                        
                    else:
                        st.write('Maaf, file tidak dapat dibaca/diproteksi')
                    # st.write(texts[:100])
                else:
                    'Belum ada file yang diupload'
    else:
        ch = st.selectbox('Pilih Perusahaan',['ACES.JK','TLKM.JK','INAF.JK','GIAA.JK','SRIL.JK','TINS.JK'])
        if ch:
            comp = yf.Ticker(ch)