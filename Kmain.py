import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate

from langchain_openai import OpenAI
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

import boto3
from langchain_community.chat_models import BedrockChat
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.runnables import RunnablePassthrough
import os
load_dotenv()

openai_api_key = st.secrets["openai_api_key"]
aws_access_key_id = st.secrets["aws_access_key_id"]
aws_secret_access_key = st.secrets["aws_secret_access_key"]

boto_session = boto3.session.Session(
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key)

#LLM부터 응답 받아오기(get response)
def get_response(query, chat_history):
    template = """
        You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question.
        정답을 모른다면 "카카오 계정 약관에 없는 내용입니다. 다시 질문해주세요."라고 해.
        Answer형식은 불렛포인트 사용하여 대답해줘.

        chat_history: {chat_history}
        context: {context}
        question: {question}
        """
    
    #prompt = ChatPromptTemplate.from_template(template)

    bedrock_runtime = boto_session.client(
        service_name="bedrock-runtime",
        region_name="us-east-1",
    )

    model_id = "anthropic.claude-3-sonnet-20240229-v1:0"

    model_kwargs =  { 
        "temperature": 0.0,
        "top_k": 250,
        "top_p": 1,
        #"stop_sequences": ["\n\nHuman"],
    }
    llm = BedrockChat(
        client=bedrock_runtime,
        model_id=model_id,
        model_kwargs=model_kwargs,
        streaming=True,
        callbacks=[StreamingStdOutCallbackHandler()]
    )

    custom_prompt = PromptTemplate(
        template=template,
        input_variables=["context", "question", "chat_history"],
    )
    #chain = prompt | llm | StrOutputParser()
    chain = ConversationalRetrievalChain.from_llm(
        llm = llm,
        verbose=True,
        retriever = load_vector_db(),
        combine_docs_chain_kwargs={"prompt": custom_prompt},
    )
    #print('chain',chain)
    # return chain.stream({
    #     "Context":load_vector_db(),
    #     "chat_history": chat_history,
    #     "user_question": query,
    # })
    return chain({"question":query, "chat_history":chat_history})
def format_docs(docs):
    # 검색한 문서 결과를 하나의 문단으로 합쳐줍니다.
    return "\n\n".join(doc.page_content for doc in docs)

def load_vector_db():
    # load db
    embeddings_model = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vectorestore = FAISS.load_local('./db/faiss', embeddings_model, allow_dangerous_deserialization=True )
    #vectorestore = FAISS.load_local('./db/faiss', embeddings_model, allow_dangerous_deserialization=True )
    retriever = vectorestore.as_retriever()
    #print('ret',retriever)
    return retriever


if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.set_page_config(page_title="RAG_DEMO", page_icon="🦜⛓️")
st.title("카카오계정 관리 약관에 대해 설명해줘")
st.title("_:orange[KAKAO]_ 계정 약관 QA BOT")

if "chat_history" not in st.session_state:
    st.session_state.chat_histroy = []

# if 'messages' not in st.session_state:
#     st.session_state['messages'] = [{"role":"assistant",
#                                      "content":"안녕하세요! 주어진 문서에 대해 질문 해 주세요."}]
#print('history:',st.session_state.chat_history)

#카카오 계정약관 목적 알려줘
#대화(conversation)      
for message in st.session_state.chat_history:
    print('msg')
    if isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.markdown(message.content)
    else:
        with st.chat_message("AI"):
            st.markdown(message.content) 
#user input
user_query = st.chat_input("질문을 입력해주세요. 예시: '카카오계정 관리 약관에 대해 설명해줘' ")
if user_query is not None and user_query != "":
    st.session_state.chat_history.append(HumanMessage(user_query))

    with st.chat_message("Human"):
        st.markdown(user_query)
    
    with st.chat_message("AI"):
        res = get_response(user_query, st.session_state.chat_history)
        ai_response = res['answer']
        # print('AI:',ai_response)
        #ai_response = st.write_stream(get_response(user_query, st.session_state.chat_history))
        #with st.spinner("답변 생성중입니다..."):
        #st.markdown(ai_response)

    st.session_state.chat_history.append(AIMessage(content=ai_response))
    st.markdown(ai_response)
    print('endthen : ', st.session_state.chat_history)
