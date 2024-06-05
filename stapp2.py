# from fastembed import TextEmbedding
import streamlit as st
import fitz
import re
import ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatOllama


st.set_page_config(page_title="ChatPDF")
st.title("Ollama Python Chatbot")

modelfile='''
FROM phi3
SYSTEM You are a financial analyst.
PARAMETER temperature 0.2
'''

ollama.create(model='phicustom', modelfile=modelfile)

def text_cleaning(text):
    text = text.lower()
    text = text.replace(r'@(\w|\d)+',' ')
    text = text.replace(r'#(\w|\d)+',' ')
    text = text.replace(r'(http|https)\S+',' ')
    text = text.replace('\n',' ')
    return text

# def embed_df(df):
#   model = TextEmbedding()
#   df['Embeddings'] = [list(model.embed(x)) for x in df['Text']]
#   return df

# # Initialize the model
# def embed_list(documents):
#     embedding_model = TextEmbedding()
#     embeddings_list = list(embedding_model.embed(documents))
#     return embeddings_list

def extractpdf(file):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    text = ''
    for page in doc:
        text += text_cleaning(page.get_text())
    return text

def reset_conversation():
  st.session_state.messages = []
  st.session_state.chat_history = []
# # Example list of documents
documents = ["",
    "This is built to be faster and lighter than other embedding libraries e.g. Transformers, Sentence-Transformers, etc.",
    "FastEmbed is supported by and maintained by Qdrant."
]

# llm = ChatOllama(model="phi3", temperature=0.2)
upl_file = st.sidebar.file_uploader('upload pdf')
if upl_file:
    tx = extractpdf(upl_file)
    # st.write(tx)
    # txlist = embed_list(list(tx))
    st.sidebar.write("file successfully uploaded")
    st.sidebar.write(tx[:20])
else:
    # txlist = embed_list(documents)
    tx = documents[0]
    st.sidebar.write("Upload file (pdf) to add context")
    
# st.sidebar.write(tx)
# initialize history
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# init models
if "model" not in st.session_state:
    st.session_state["model"] = ""

models = [model["name"] for model in ollama.list()["models"]]
st.session_state["model"] = st.selectbox("Choose your model", models)

def model_res_generator():
    stream = ollama.chat(
        # model=st.session_state["model"],
        model = 'phicustom',
        messages=st.session_state["messages"],
        # messages=context,
        stream=True,
    )
    # stream = llm.invoke(st.session_state["messages"])
    for chunk in stream:
        yield chunk["message"]["content"]

# Display chat messages from history on app rerun
# for message in st.session_state["messages"]:
#     with st.chat_message(message["role"]):
#         st.markdown(message["content"])



if prompt := st.chat_input("What is up?"):
    # add latest message to history in format {role, content}
    st.session_state["messages"].append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        # st.markdown(prompt)
        question = prompt
        prompt = ChatPromptTemplate.from_template(
            f"""
            <s> [INST] You are a financial assistant for question-answering tasks. 
            Answer the user question based on provided context. If you don't know the answer, just say that you don't know. [/INST] </s> 
            [INST] Question: {question} 
            Context: {tx}
            Answer: [/INST]
            """
        )
        st.markdown(question)

    with st.chat_message("assistant"):
        st.write(prompt)
        message = st.write_stream(model_res_generator())
        st.session_state["messages"].append({"role": "assistant", "content": message})

st.button('Reset Chat', on_click=reset_conversation)
# from langchain_community.llms import Ollama 
# import pandas as pd
# from pandasai import SmartDataframe
# # from pandasai.llm.starcoder import Starcoder

# llm = Ollama(model="llama3")
# # llm = Starcoder(api_token=API_KEY)

# st.title("Data Analysis with PandasAI")

# uploader_file = st.file_uploader("Upload a CSV file", type= ["csv"])

# if uploader_file is not None:
#     data = pd.read_csv(uploader_file)
#     st.write(data.head(3))
#     df = SmartDataframe(data, config={"llm": llm})
#     prompt = st.text_area("Enter your prompt:")

#     if st.button("Generate"):
#         if prompt:
#             with st.spinner("Generating response..."):
#                 st.write(df.chat(prompt))
#         else:
#             st.warning("Please enter a prompt!")

# import streamlit.components.v1 as stc 
# import pandas as pd 
# import pygwalker as pyg 
# from pygwalker.api.streamlit import StreamlitRenderer

# @st.cache_data
# def getData(url,sep):
#     df = pd.read_csv(url,sep=sep)
#     return df
# with st.form("upload_form"):
#     data_file = st.file_uploader("Upload a CSV File",type=["csv","txt"])
#     submitted = st.form_submit_button("Submit")

#     if submitted:
#         # df = pd.read_csv(data_file)
#         df = getData(data_file,',')
#         st.dataframe(df.head())
#         pyg_app = StreamlitRenderer(df)
#         pyg_app.explorer()

