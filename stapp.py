import streamlit as st
from hugchat import hugchat
from hugchat.login import Login
# set browser.gatherUsageStats to false

EMAIL = 'agung.septia@gmail.com'
PASS = 'FXqk*4REQ9/9)d;'
hf_email = EMAIL
hf_pass = PASS
sign = Login(hf_email, hf_pass)
cookies = sign.login()
chatbot = hugchat.ChatBot(cookies=cookies.get_dict())
# Function for generating LLM response
def generate_response(prompt_input):
    return chatbot.chat(prompt_input)

# menu = ["mistral","phi3"]
# choice = st.sidebar.selectbox("Select Menu", menu)

# st.write(chatbot.get_available_llm_models())  
         
chatbot.switch_llm(0)

# if choice=="mistral":
    st.title('AI-Financial Assistant')
    st.text('Dibangun dengan Model LLM Huggingface yang dapat menjadi alternative open source (gratis) dari ChatGPT')
    st.text('sumber: https://huggingface.co/chat/')
    # App title
    # st.set_page_config(page_title="🤗💬 HugChat")
    # Store LLM generated responses
    if "messages" not in st.session_state.keys():
        st.session_state.messages = [{"role": "assistant", "content": "Halo, silakan bertanya :)"}]

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])


    # User-provided prompt
    if prompt := st.chat_input(disabled=not (hf_email and hf_pass)):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

    # Generate a new response if last message is not from assistant
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = generate_response(prompt) 
                st.write(response) 
        message = {"role": "assistant", "content": response}
        st.session_state.messages.append(message)

elif choice=="phi3":
    st.write("blank")
