def DSRI(df,y1,y0):
    return (df.at["Accounts Receivables", y1] / df.at["Revenue", y1]) / (df.at["Accounts Receivables", y0] / df.at["Revenue", y0])
def GMI(df,y1,y0):
    return ((df.at["Revenue", y0] - df.at["Cost of Goods Sold", y0])/df.at["Revenue", y0]) / ((df.at["Revenue", y1] - df.at["Cost of Goods Sold", y1])/df.at["Revenue", y1])

def AQI(df,y1,y0):
    AQI_t1 = (1 - (df.at["Current Assets", y1] +
              df.at["Property, Plant & Equipment", y1]+df.at["Securities", y1])) / df.at["Total Assets", y1]
    AQI_t2 = (1 - (df.at["Current Assets", y0] +
              df.at["Property, Plant & Equipment", y0]+df.at["Securities", y0])) / df.at["Total Assets", y0]
    return AQI_t1 / AQI_t2

def SGI(df,y1,y0):
    return (df.at["Revenue", y1] / df.at["Revenue", y0])

def DEPI(df,y1,y0):
    DEPI_t1 = (df.at["Depreciation", y0] / (df.at["Depreciation",
               y0] + df.at["Property, Plant & Equipment", y0]))
    DEPI_t2 = (df.at["Depreciation", y1] / (df.at["Depreciation",
               y1] + df.at["Property, Plant & Equipment", y1]))
    return DEPI_t1 / DEPI_t2

def SGAI(df,y1,y0):
    return (df.at["Selling, General & Admin.Expense", y1] / df.at["Revenue", y1]) / (df.at["Selling, General & Admin.Expense", y0] / df.at["Revenue", y0])

def LVGI(df,y1,y0):
    return ((df.at["Current Liabilities", y1] + df.at["Total Long-term Debt", y1]) / df.at["Total Assets", y1]) / ((df.at["Current Liabilities", y0] + df.at["Total Long-term Debt", y0]) / df.at["Total Assets", y0])

def TATA(df,y1,y0):
    return (df.at["Net Income from Continuing Operations", y1] - df.at["Cash Flow from Operations", y1]) / df.at["Total Assets", y1]

def BeneishMScore(dsri, gmi, aqi, sgi, depi, sgai, lvgi, tata):
    return -4.84+0.92*dsri+0.528*gmi+0.404*aqi+0.892*sgi+0.115*depi-0.172*sgai+4.679*tata-0.327*lvgi

def SUSTAX(df,y1,y0):
    return (df.at["Selling And Marketing Expense", y1] - df.at["Selling And Marketing Expense", y0])/(df.at["Tax Provision", y1]-df.at["Tax Provision", y0])


import fitz
import re
from collections import Counter
import pymupdf
import pandas as pd
fitz.TOOLS.set_small_glyph_heights(True)

def text_cleaning(text):
    text = text.lower()
    text = text.replace(r'@(\w|\d)+',' ')
    text = text.replace(r'#(\w|\d)+',' ')
    text = text.replace(r'(http|https|www)\S+',' ')
    text = text.replace(r'\d+',' ')
    from string import punctuation
    for punct in punctuation:
        text = text.replace(punct, ' ')
    text = text.replace('\n',' ')
    return text

def readpdf(pdfpath):
  doc = pymupdf.open(stream=pdfpath.read(),filetype="pdf")
  texts = ''
  for page in doc:
      texts += text_cleaning(page.get_text())
  return texts

lmidx = pd.read_csv('lmdict_index.csv',sep=";")
lmlist =['negative','positive','uncertainty','litigious','strong_modal','weak_modal',
    'constraining', 'complexity']
    #    , 'anger', 'anticipation', 'disgust',
    #    'fear', 'joy', 'negative_alt', 'positive_alt', 'sadness', 'surprise',
    #    'trust']
lm_neg = lmidx[lmidx[lmlist[0]] ==1]
lm_pos = lmidx[lmidx[lmlist[1]] ==1]
lm_unc = lmidx[lmidx[lmlist[2]] ==1]
lm_lit = lmidx[lmidx[lmlist[3]] ==1]
lm_stm = lmidx[lmidx[lmlist[4]] ==1]
lm_wkm = lmidx[lmidx[lmlist[5]] ==1]
lm_con = lmidx[lmidx[lmlist[6]] ==1]
lm_com = lmidx[lmidx[lmlist[7]] ==1]
list_lm = [lm_neg,lm_pos,lm_unc,lm_lit,lm_stm,lm_wkm,lm_con,lm_com]

def countLm(text,ll):
  cdict = {}
  words = re.findall(r"[^\W_]+", text, re.MULTILINE)
  c = Counter(words)
  for x in ll:
    if c[x]>0:
      cdict[x]=c[x]
    else:
      pass
  # return [sum(cdict.values()),cdict]
  return sum(cdict.values())

def getLm(text,ll):
  cdict = {}
  words = re.findall(r"[^\W_]+", text, re.MULTILINE)
  c = Counter(words)
  for x in ll:
    if c[x]>0:
      cdict[x]=c[x]
    else:
      pass
  # return [sum(cdict.values()),cdict]
  return cdict

import numpy as np
def assignLm(df):
  for x in range(len(lmlist)):
    ll = list_lm[x]['lm_indo']
    df['totalwords'] = [len(re.findall(r"[^\W_]+", i, re.MULTILINE)) for i in df['text']]
    df[lmlist[x]] = df.apply(lambda x: countLm(x['text'], ll),axis=1)
    # df['count_'+str(lmlist[x])] = [i[0] for i in df[lmlist[x]]]
    # df[lmlist[x]] = [i[1] for i in df[lmlist[x]]]
    df = df.replace('',np.nan).fillna(0)
  return df

def getwordLm(df):
  for x in range(len(lmlist)):
    ll = list_lm[x]['lm_indo']
    df['totalwords'] = [len(re.findall(r"[^\W_]+", i, re.MULTILINE)) for i in df['text']]
    df[lmlist[x]] = df.apply(lambda x: getLm(x['text'], ll),axis=1)
    # df['count_'+str(lmlist[x])] = [i[0] for i in df[lmlist[x]]]
    # df[lmlist[x]] = [i[1] for i in df[lmlist[x]]]
    df = df.replace('',np.nan).fillna(0)
  return df

import joblib
def get_prediction(model_url,df):
  loaded_model = joblib.load(model_url)
  result = loaded_model.predict(df)
  prob = loaded_model.predict_proba(df)
  return [result,prob]

import yfinance as yf
from yahooquery import Ticker

def getBenData(tick,year):
  comp = yf.Ticker(tick)
  y0 = str(year-1)
  y1 = str(year)
  balanceSheet = comp.balancesheet
  incomeStatement = comp.financials
  cashFlow = comp.cashflow

  # Balance Sheet
  balanceSheet = balanceSheet[balanceSheet.columns[0:2]]
  balanceSheet.columns = [y1, y0]
  balanceSheet = balanceSheet.fillna(0).astype(float)

  # Income Statement
  incomeStatement = incomeStatement[incomeStatement.columns[0:2]]
  incomeStatement.columns = [y1, y0]
  incomeStatement = incomeStatement.fillna(0).astype(float)


  # Cash Flow
  cashFlow = cashFlow[cashFlow.columns[0:2]]
  cashFlow.columns = [y1, y0]
  cashFlow.dropna()

  cogs23 = incomeStatement.at['Total Revenue', y1] - \
      incomeStatement.at['Gross Profit', y1]
  cogs22 = incomeStatement.at['Total Revenue', y0] - \
      incomeStatement.at['Gross Profit', y0]

  # COGS = pd.Series(data={'2023': cogs23, '2022': cogs22},
  #                  name='Cost Of Revenue')
  incomeStatement.loc['Cost Of Revenue'] = [cogs23, cogs22]

  # long term Debt

  if('Long Term Debt' not in balanceSheet.index):
      ld23 = balanceSheet.at['Long Term Debt And Capital Lease Obligation', "2023"]
      ld22 = balanceSheet.at['Long Term Debt And Capital Lease Obligation', "2022"]
  else:
      ld23 = balanceSheet.at['Long Term Debt', y1]
      ld22 = balanceSheet.at['Long Term Debt', y0]
  balanceSheet.loc['Long Term Debt'] = [ld23, ld22]


  if('Long Term Investments' not in balanceSheet.index):
      li23 = balanceSheet.at['Long Term Equity Investment', y1]
      li22 = balanceSheet.at['Long Term Equity Investment', y0]
  else:
      li23 = balanceSheet.at['Long Term Investment', y1]
      li22 = balanceSheet.at['Long Term Investment', y0]

  balanceSheet.loc['Long Term Investments'] = [li23, li22]

  if('Receivables' not in balanceSheet.index):
      li23 = balanceSheet.at['Accounts Receivable', y1]
      li22 = balanceSheet.at['Accounts Receivable', y0]
  else:
      li23 = balanceSheet.at['Receivables', y1]
      li22 = balanceSheet.at['Receivables', y0]

  balanceSheet.loc['Receivables'] = [li23, li22]

  # Extracting the statements

  df1 = incomeStatement.loc[["Total Revenue",
                          "Cost Of Revenue", "Selling General And Administration", "Net Income Continuous Operations"]]

  df2 = balanceSheet.loc[["Receivables",
                          "Current Assets", "Gross PPE", "Long Term Investments", "Total Assets", "Current Liabilities", "Long Term Debt"]]
  df3 = incomeStatement.loc[["Depreciation And Amortization In Income Statement"]]
  df4 = cashFlow.loc[["Receiptsfrom Customers"]]
  df5 = incomeStatement.loc[['Selling And Marketing Expense','Tax Provision']]

  df = pd.concat([df1, df2, df3,df4,df5])
  df = df.reindex(['Total Revenue', 'Cost Of Revenue', 'Selling General And Administration', 'Depreciation And Amortization In Income Statement', 'Net Income Continuous Operations', 'Receivables', 'Current Assets',
                      'Gross PPE', 'Long Term Investments', 'Total Assets', 'Current Liabilities', 'Long Term Debt', 'Receiptsfrom Customers',
                      'Selling And Marketing Expense','Tax Provision'])
  df.index = ["Revenue", "Cost of Goods Sold", "Selling, General & Admin.Expense", "Depreciation", "Net Income from Continuing Operations", "Accounts Receivables",
              "Current Assets", "Property, Plant & Equipment", "Securities", "Total Assets", "Current Liabilities", "Total Long-term Debt", "Cash Flow from Operations",
              'Selling And Marketing Expense','Tax Provision'] #tambahan SUSTAX

  dsri = DSRI(df,y1,y0)
  gmi = GMI(df,y1,y0)
  aqi = AQI(df,y1,y0)
  sgi = SGI(df,y1,y0)
  depi = DEPI(df,y1,y0)
  sgai = SGAI(df,y1,y0)
  lvgi = LVGI(df,y1,y0)
  tata = TATA(df,y1,y0)
  data2 = {"COMP":tick,
           "YEAR":year,
           "DSRI":dsri,
            "GMI":gmi,
            "AQI":aqi,
            "SGI":sgi,
            "DEPI":depi,
            "SGAI":sgai,
            "LVGI":lvgi,
            "TATA":tata,
            "BMscore":BeneishMScore(dsri, gmi, aqi, sgi, depi, sgai, lvgi, tata)
        }
  # ratios = pd.DataFrame(data2)
  return [df,data2]

import pandas as pd
def getBenDF(complist,yr):
  df2 = None
  for x in range(len(complist)):
    # comp = yf.Ticker(complist[x])
    data = []
    try:
      bmdict = getBenData(complist[x],yr)
      data.append(bmdict)
      df1 = pd.DataFrame(data)
      df2 = pd.concat([df2,df1],ignore_index=True).drop_duplicates().reset_index(drop=True)
    except:
      pass
  return df2


from wordcloud import WordCloud
import json
from urllib.request import urlopen

import nltk
# nltk.download('punkt')
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
import plotly_express as px

def create_wordcloud(datadict):
    # word_could_dict = {'Git':100, 'GitHub':100, 'push':50, 'pull':10, 'commit':80, 'add':30, 'diff':10, 
    #               'mv':5, 'log':8, 'branch':30, 'checkout':25}
    wordcloud = WordCloud(width = 1000, height = 500)\
                .generate_from_frequencies(datadict)

    # plt.figure(figsize=(15,8))
    # plt.imshow(wordcloud)
    fig = px.imshow(wordcloud)
    fig.update_yaxes(title=None, visible=False)
    fig.update_xaxes(title=None, visible=False)
    return fig

from functools import reduce
def merge_dictionaries(dlist):
  merged_dict = dlist[0].copy()
  for d in range(len(dlist)-1):
    merged_dict.update(dlist[d+1])
  return merged_dict