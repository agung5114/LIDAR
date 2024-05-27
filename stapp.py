import streamlit as st

menu = ["mistral","phi3"]
choice = st.sidebar.selectbox("Select Menu", menu)

if choice=="mistral":
    from hugchat import hugchat
    from hugchat.login import Login

    st.subheader('Artificial Intelligence Smart Assistant (AISA)')
    st.text('Dibangun dengan Model HuggingChat dari Huggingface yang dapat menjadi alternative open source (gratis) dari ChatGPT')
    st.text('sumber: https://huggingface.co/chat/')
    # App title
    # st.set_page_config(page_title="🤗💬 HugChat")
    EMAIL = 'agung.septia@gmail.com'
    PASS = 'FXqk*4REQ9/9)d;'
    hf_email = EMAIL
    hf_pass = PASS
    # Store LLM generated responses
    if "messages" not in st.session_state.keys():
        st.session_state.messages = [{"role": "assistant", "content": "Saya AISA, silakan bertanya :)"}]

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Function for generating LLM response
    def generate_response(prompt_input, email, passwd):
        # Hugging Face Login
        sign = Login(email, passwd)
        cookies = sign.login()
        # Create ChatBot                        
        chatbot = hugchat.ChatBot(cookies=cookies.get_dict())
        return chatbot.chat(prompt_input)

    # User-provided prompt
    if prompt := st.chat_input(disabled=not (hf_email and hf_pass)):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

    # Generate a new response if last message is not from assistant
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = generate_response(prompt, hf_email, hf_pass) 
                st.write(response) 
        message = {"role": "assistant", "content": response}
        st.session_state.messages.append(message)

elif choice=="phi3":
    st.write("blank")
