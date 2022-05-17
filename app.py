import streamlit as st
import json
import chat_utils
from annotated_text import annotated_text

def get_text():
    input_text = st.text_input("You: ","", key="input", placeholder ="How are you?")
    _ = st.text_input("You: ","", key="input", placeholder =input_text)
    return input_text
def message(msg,user, order):
    col1, col12,col2,col3 = st.columns([2,6, 1,8])
    if user == 'bot':
        with col2:
            st.image("sea_slug.png", width =50)
        with col3:
            if order == 1:
                annotated_text(("", msg , "#8ef"))
            else:
                st.write(msg)
    else:
        with col1:
            st.markdown(" You :smile: "+": ")
        with col12:
            if order == 1:
                annotated_text(("", msg, "#8ef"))
            else:
                st.markdown(msg)

st.title('PyTorch Chat Bot')

bot = st.sidebar.selectbox(
    'Select a bot',
    ('sea slug', 'cat')
)

if 'generated' not in st.session_state:
    st.session_state['generated'] = []

if 'past' not in st.session_state:
    st.session_state['past'] = []


model_path = "data.pth"
model, all_words, tags = chat_utils.load_model(model_path)

#st.write('### Bot says:')

if bot:
    message(msg ='### Bot says: I am Sam the sea slug!', user = 'bot', order = 10)
    #st.write('### Bot says: I am Sam the sea slug!')

    you_say = get_text()
    if you_say:
        bot_say = chat_utils.chatting(model,you_say, all_words, tags)
        st.session_state.past.append(you_say)
        st.session_state.generated.append(bot_say)
    if st.session_state['generated']:
        for i in range(len(st.session_state['generated']) - 1, -1, -1):
            #if i == len(st.session_state['generated'])-1:
            #    msg = annotated_text(("",st.session_state["past"][i],"#8ef"))
            #else:
            #    msg = st.session_state["past"][i]
            #message(msg, user='you', len(st.session_state['generated'])-i)
            message(st.session_state["past"][i],user = 'you', order =len(st.session_state['generated']) - i)
            message(st.session_state["generated"][i], user='bot', order =len(st.session_state['generated']) - i)
