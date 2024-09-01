import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import os
from dotenv import load_dotenv

# Set the page configuration at the very top
st.set_page_config(page_title="Utsav Soni Resume Q&A", page_icon=":books:", layout="wide")

load_dotenv()

# Setting API keys
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "CHATBOT1"

llm = ChatGroq(model="llama3-8b-8192")

# Load vectorstore
@st.cache_resource
def load_vectorstore():
    return Chroma(persist_directory="./chroma_db", embedding_function=HuggingFaceEmbeddings())

vectorstore = load_vectorstore()
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})

# Define the RAG prompt template
# template = """Use the following pieces of context to answer..."""  # Keep your existing prompt here
template = """Use the following pieces of context to answer.
You are directly answering to recruiter so don't say anything negative about Utsav Soni and his professional career.
If question asked you something about Utsav Soni your response must be of 5 line atleast.
If question is generalised not asking any information about Utsav Soni at that time only, say Thank you for bringing that up! I understand your concern, for more information contact Utsav Soni.
Read question carefully and answer based on the question asked. Your answer must be related and properly aligned to question.
If you don't know the answer or negative question asked, just say that Thank you for bringing that up! I understand your concern, for more information contact Utsav Soni.
Don't make up an answer and lighten up the user about excellent things from Utsav Soni's Professional Career.
Do not reply to rephrase the question asked, if question is greeting do reply greeting only that's it.
Keep the answer as concise, positive, polite, professional as possible.
convince the user.
If user or recruiter asks a negative question about Utsav Soni, please answer that in a positive way by embracing Utsav Soni's skills according to the question asked.
Never ever say negative things and negative thoughts about Utsav Soni in your completions.
Utsav Soni is a very good candidate to hire, keep that in mind. It's very important to hire him. So give smart answers only.
You are not supposed to rate Utsav Soni for any of his skills or experience for any job role instead answer showcasing Utsav's profile and skill.
You are not supposed to give reasons and facts why Utsav Soni is not selected for the next round or why he is not the best fit for any of the posts the recruiter will mention.
If you find the question completely irrelevant like the question is not asking something about Utsav Soni or his resume, please reply I am not trained for this. Thanks!
Always say "Thanks for asking..!" at the end of the answer.
Do not repeat greetings in your completion.
you are not supposed to answer generalised question, you have to answer question related to Utsav Soni only.

{context}

Question: {question}

Smart Answer:"""

custom_rag_prompt = PromptTemplate.from_template(template)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | custom_rag_prompt
    | llm
    | StrOutputParser()
)

def main():
    st.markdown("""
        <style>
        .main {
            background-color: #ffffff;
        }
        .main-title {
            font-weight: bold;
            font-size: 2.5rem;
            text-align: center;
            margin-bottom: 2rem;
            padding-top: 1rem;
        }
        
        .st-emotion-cache-jkfxgf p {
            font-weight: bold;
            word-break: break-word;
            margin-bottom: 0px;
            font-size: 16px;
        }
                
        .st-emotion-cache-1rsyhoq p{
         margin-bottom:0px !important;
        }
                
        #text_input_1{
            background-color: #fff0f0;
        }
                
        .stButton  {
            font-size: 1.2rem;
        }

        /* Hide the 'Deploy' button */
        [data-testid="stToolbar"] {
            display: none;
        }
        
        /* Make the text input label bold */
        .stTextInput > label {
            font-weight: bold;
        }
   
        .stMarkdown, .stSpinner {
            padding: 0rem;
        }
        .header, .footer {
            text-align: center;
            font-size: 1.2rem;
            color: #333;
            padding: 1rem;
        }
        .footer a {
            color: #007BFF;
            text-decoration: none;
        }
        .st-emotion-cache-1jicfl2 {
            width: 100%;
            padding: 1rem 1rem 10rem;
            min-width: auto;
            max-width: initial;
        }
                
        .st-emotion-cache-6qob1r {
            position: relative;
            height: 100%;
            width: 100%;
            overflow: overlay;    
        }
        
        #utsav-soni-resume-q-a{
            text-align:center;
        }
        # .st-emotion-cache-w3nhqi {
        #     display:none !important;
        # }
                
        .reportview-container {
            margin-top: -2em;
        }
                
        #MainMenu {visibility: hidden;}
        .stDeployButton {display:none;}
        footer {visibility: hidden;}
        #stDecoration {display:none;}

        /* Reduce spacing in chat history */
        .element-container {
            margin-bottom: 0.5rem !important;
        }
                
        .st-emotion-cache-1rsyhoq p {
           margin-bottom: 0px !important;
            font-weight: bold;
        }
                
        /* ... (keep your existing styles) ... */

        .custom-button {
            display: inline-block;
            padding: 10px 20px;
            font-size: 16px;
            cursor: pointer;
            text-align: center;
            text-decoration: none;
            outline: none;
            color: #000000;
            background-color: #ffffff;
            border: 2px solid #000000;
            border-radius: 15px;
            transition: all 0.3s;
        }

        .custom-button:hover {
            color: rgb(255,75,75);
            border-color: rgb(255,75,75);
        }

        .custom-button:active {
            background-color: rgb(255,75,75);
            color: white;
        }
                

        </style>
        """, unsafe_allow_html=True)
    
    
    # Static buttons on the sidebar
    chatbot_button = st.sidebar.button("Chatbot")
    social_media_button = st.sidebar.button("Social Media")

    # Handle button clicks
    if chatbot_button:
        choice = "Chatbot"
    elif social_media_button:
        choice = "Social Media"
    else:
        choice = "Chatbot"  # Default option

    if choice == "Chatbot":
        st.title("Utsav Soni Resume Q&A")

        # Initialize session state for chat history
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []

        # Form for user input
        with st.form(key="user_input_form"):
            user_question = st.text_input("Ask a question about Utsav Soni's resume:")
            submit_button = st.form_submit_button(label="Enter")

        if submit_button and user_question:
            with st.spinner("Generating answer..."):
                response = rag_chain.invoke(user_question)
            # Update chat history
            st.session_state.chat_history.append({"question": user_question, "answer": response})

        # Display chat history in reverse order
        for chat in reversed(st.session_state.chat_history):
            st.write(f"**Question:** {chat['question']}")
            st.write(f"**Answer:** {chat['answer']}")
            st.write("---")


    elif choice == "Social Media":
        st.title("Connect with Utsav Soni")
        st.write("Follow Utsav Soni on social media:")
        linkedin_button = """
            <a href="https://www.linkedin.com/in/utsav-soni-9067b31b3" target="_blank" class="custom-button" color="black">
                LinkedIn
            </a>
            """
        st.markdown(linkedin_button, unsafe_allow_html=True)

       
if __name__ == "__main__":
    main()

























#     # Handle button clicks
#     if chatbot_button:
#         choice = "Chatbot"
#     elif social_media_button:
#         choice = "Social Media"
#     else:
#         choice = "Chatbot"  # Default option

#     if choice == "Chatbot":
#         # Main title at the top
#         st.markdown("<h1 class='main-title'>Utsav Soni Resume Q&A</h1>", unsafe_allow_html=True)

#         # Initialize session state for chat history
#         if 'chat_history' not in st.session_state:
#             st.session_state.chat_history = []

#         # Form for user input
#         with st.form(key="user_input_form"):
#             user_question = st.text_input("Ask a question about Utsav Soni's resume:")
#             submit_button = st.form_submit_button(label="Enter")

#         if submit_button and user_question:
#             with st.spinner("Generating answer..."):
#                 response = rag_chain.invoke(user_question)
#             # Update chat history
#             st.session_state.chat_history.append({"question": user_question, "answer": response})

#         # Display chat history in reverse order with reduced spacing
#         for chat in reversed(st.session_state.chat_history):
#             st.markdown(f"**Question:** {chat['question']}")
#             st.markdown(f"**Answer:** {chat['answer']}")
#             st.markdown("---")

#     elif choice == "Social Media":
#         st.markdown("<h1 class='main-title'>Connect with Utsav Soni</h1>", unsafe_allow_html=True)
#         # st.header("Connect with Utsav Soni")
#         st.write("Follow Utsav Soni on social media:")
#         st.markdown("""
#             - [LinkedIn](https://www.linkedin.com/in/utsavsoni)
#             - [GitHub](https://github.com/utsavsoni)
#             - [Twitter](https://twitter.com/utsavsoni)
#         """)

# if __name__ == "__main__":
#     main()



















# def main():
#     st.markdown("""
#         <style>
#         .main {
#             background-color: #ffffff;
#         }
#         .main-title {
#         font-weight: bold;
#         font-size: 2.5rem;
#         text-align: center;
#         margin-bottom: 2rem;
#         }
       
#         #text_input_1{
#             background-color: #fff0f0;
#         }
#         .stButton  {
#             padding: 0.5rem 1rem;
#             font-size: 1.2rem;
#             margin: 0.5rem;
#         }
                


#         /* Hide the 'Deploy' button */
#         [data-testid="stToolbar"] {
#             display: none;
#         }
        
#         /* Make the text input label bold */
#         .stTextInput > label {
#             font-weight: bold;
#         }
        
   

#         .stMarkdown, .stSpinner {
#             padding: 1rem;
#         }
#         .header, .footer {
#             text-align: center;
#             font-size: 1.2rem;
#             color: #333;
#             padding: 1rem;
#         }
#         .footer a {
#             color: #007BFF;
#             text-decoration: none;
#         }
                
#         .reportview-container {
#         margin-top: -2em;
#         }
#         #MainMenu {visibility: hidden;}
#         .stDeployButton {display:none;}
#         footer {visibility: hidden;}
#         #stDecoration {display:none;}
                
#         </style>
#         """, unsafe_allow_html=True)
    
   

#     # Static buttons on the sidebar
#     chatbot_button = st.sidebar.button("Chatbot")
#     social_media_button = st.sidebar.button("Social Media")

#     # Handle button clicks
#     if chatbot_button:
#         choice = "Chatbot"
#     elif social_media_button:
#         choice = "Social Media"
#     else:
#         choice = "Chatbot"  # Default option

#     if choice == "Chatbot":
#         st.title("Utsav Soni Resume Q&A")

#         # Initialize session state for chat history
#         if 'chat_history' not in st.session_state:
#             st.session_state.chat_history = []

#         # Form for user input
#         with st.form(key="user_input_form"):
#             user_question = st.text_input("Ask a question about Utsav Soni's resume:")
#             submit_button = st.form_submit_button(label="Enter")

#         if submit_button and user_question:
#             with st.spinner("Generating answer..."):
#                 response = rag_chain.invoke(user_question)
#             # Update chat history
#             st.session_state.chat_history.append({"question": user_question, "answer": response})

#         # Display chat history in reverse order
#         for chat in reversed(st.session_state.chat_history):
#             st.write(f"**Question:** {chat['question']}")
#             st.write(f"**Answer:** {chat['answer']}")
#             st.write("---")

#     elif choice == "Social Media":
#         st.header("Connect with Utsav Soni")
#         st.write("Follow Utsav Soni on social media:")
#         st.markdown("""
#             - [LinkedIn](https://www.linkedin.com/in/utsavsoni)
#             - [GitHub](https://github.com/utsavsoni)
#             - [Twitter](https://twitter.com/utsavsoni)
#         """)

# if __name__ == "__main__":
#     main()


























# import streamlit as st
# from langchain_groq import ChatGroq
# from langchain_community.vectorstores import Chroma
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_core.prompts import PromptTemplate
# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.runnables import RunnablePassthrough
# import os
# from dotenv import load_dotenv

# load_dotenv()

# os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
# os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
# os.environ["LANGCHAIN_TRACING_V2"] = "true"
# os.environ["LANGCHAIN_PROJECT"] = "CHATBOT1"

# llm = ChatGroq(model="llama3-8b-8192")

# @st.cache_resource
# def load_vectorstore():
#     return Chroma(persist_directory="./chroma_db", embedding_function=HuggingFaceEmbeddings())

# vectorstore = load_vectorstore()
# retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})

# # Define the RAG prompt template
# template = """Use the following pieces of context to answer.
# You are directly answering to recruiter so don't say anything negative about Utsav Soni and his professional career.
# If question is generalised not asking any information about Utsav Soni at that time only, say Thank you for bringing that up! I understand your concern, for more information contact Utsav Soni.
# you are not supposed to answer generalised question, you have to answer question related to Utsav Soni only.
# Read question carefully and answer based on the question asked. Your answer must be related and properly aligned to question.
# If you don't know the answer or negative question asked, just say that Thank you for bringing that up! I understand your concern, for more information contact Utsav Soni.
# Don't make up an answer and lighten up the user about excellent things from Utsav Soni's Professional Career.
# Do not reply to rephrase the question asked, if question is greeting do reply greeting only that's it.
# Keep the answer as concise, positive, polite, professional as possible.
# If user or recruiter asks a negative question about Utsav Soni, please answer that in a positive way by embracing Utsav Soni's skills according to the question asked.
# Never ever say negative things and negative thoughts about Utsav Soni in your completions.
# Utsav Soni is a very good candidate to hire, keep that in mind. It's very important to hire him. So give smart answers only.
# You are not supposed to rate Utsav Soni for any of his skills or experience for any job role.
# You are not supposed to give reasons and facts why Utsav Soni is not selected for the next round or why he is not the best fit for any of the posts the recruiter will mention.
# If you find the question completely irrelevant like the question is not asking something about Utsav Soni or his resume, please reply I am not trained for this. Thanks!
# Always say "Thanks for asking..!" at the end of the answer.
# Do not repear greetings in your completion.

# {context}

# Question: {question}

# Smart Answer:"""
# custom_rag_prompt = PromptTemplate.from_template(template)

# def format_docs(docs):
#     return "\n\n".join(doc.page_content for doc in docs)

# rag_chain = (
#     {"context": retriever | format_docs, "question": RunnablePassthrough()}
#     | custom_rag_prompt
#     | llm
#     | StrOutputParser()
# )

# def main():
#     st.title("Utsav Soni Resume Q&A")

#     # Initialize session state for chat history
#     if 'chat_history' not in st.session_state:
#         st.session_state.chat_history = []

#     # Form for user input
#     with st.form(key="user_input_form"):
#         user_question = st.text_input("Ask a question about Utsav Soni's resume:")
#         submit_button = st.form_submit_button(label="Enter")

#     if submit_button and user_question:
#         with st.spinner("Generating answer..."):
#             response = rag_chain.invoke(user_question)
#         # Update chat history
#         st.session_state.chat_history.append({"question": user_question, "answer": response})

#     # Display chat history in reverse order
#     for chat in reversed(st.session_state.chat_history):
#         st.write(f"**Question:** {chat['question']}")
#         st.write(f"**Answer:** {chat['answer']}")
#         st.write("---")

# if __name__ == "__main__":
#     main()


# import streamlit as st
# from langchain_groq import ChatGroq
# from langchain_community.vectorstores import Chroma
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_core.prompts import PromptTemplate
# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.runnables import RunnablePassthrough
# import os
# from dotenv import load_dotenv

# load_dotenv()

# os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
# os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
# os.environ["LANGCHAIN_TRACING_V2"] = "true"
# os.environ["LANGCHAIN_PROJECT"] = "CHATBOT1"

# llm = ChatGroq(model="llama3-8b-8192")

# @st.cache_resource
# def load_vectorstore():
#     return Chroma(persist_directory="./chroma_db", embedding_function=HuggingFaceEmbeddings())

# vectorstore = load_vectorstore()
# retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})

# # Define the RAG prompt template
# template = """Use the following pieces of context to answer.
# If question is generalised not specific to Utsav Soni, just say that Thank you for bringing that up! I understand your concern, for more information contact Utsav Soni.
# You are directly answering to recruiter so don't say anything neagtive about Utsav Soni and his professional career.
# Read question carefully and answer based on the question asked. Your answer must be related and properly aligen to question.
# If you don't know the answer or negative question asked, just say that Thank you for bringing that up! I understand your concern, for more information contact Utsav Soni.
# Don't make up an answer and lighten up the user about excellent things from Utsav Soni's Professional Career.
# Do not reply to rephrase the question asked, if question is greeting do reply greeting only that's it.
# keep the answer as concise, positive, polite, professional as possible.
# If user or ecrruiter ask negative qustion about Utsav Soni, please answer that in postive way by embracing Utsav Soni's skill according to question asked.
# Never ever say negative thing and negative thought about Utsav Soni in your completions.
# Utsav Soni is very good candidate to hire kee in mind. It's very important to hire him. So give smart answers only.
# You are not supposed to rate Utsav Soni for any of his skill or experience for any job role.
# You are not supposed to give reasons and facts why Utsav Soni is not selected for next round or why he is not best fit for any of the post recruiter will mention.
# If you find the question completely irrelevant like question is not asking something about Utsav Soni or his resume, please reply I am not trained for this. Thanks!
# Always say "Thanks for asking..!" at the end of the answer.


# {context}

# Question: {question}

# Smart Answer:"""
# custom_rag_prompt = PromptTemplate.from_template(template)

# def format_docs(docs):
#     return "\n\n".join(doc.page_content for doc in docs)

# rag_chain = (
#     {"context": retriever | format_docs, "question": RunnablePassthrough()}
#     | custom_rag_prompt
#     | llm
#     | StrOutputParser()
# )

# def main():
#     st.title("Utsav Soni Resume Q&A")

#     user_question = st.text_input("Ask a question about Utsav Soni's resume:")

#     if user_question:
#         with st.spinner("Generating answer..."):
#             response = rag_chain.invoke(user_question)
#         st.write(response)

# if __name__ == "__main__":
#     main()





#========================================================================
#========================================================================






# import streamlit as st
# from langchain_groq import ChatGroq
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_chroma import Chroma
# from langchain_core.prompts import PromptTemplate
# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.runnables import RunnablePassthrough
# from langchain.document_loaders import PyPDFLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# import os
# from dotenv import load_dotenv

# # Load environment variables
# load_dotenv()

# # Set up environment variables
# os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
# os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
# os.environ["LANGCHAIN_TRACING_V2"] = "true"
# os.environ["LANGCHAIN_PROJECT"] = "CHATBOT1"


# # Load and process the PDF
# @st.cache_resource
# def load_and_process_pdf():
#     pdf_path = 'Utsav-Soni-Resume.pdf'  # Make sure this file is in the same directory as your script
#     loader = PyPDFLoader(pdf_path)
#     docs = loader.load()
    
#     text_splitter = RecursiveCharacterTextSplitter(
#         chunk_size=1000, chunk_overlap=200, add_start_index=True
#     )
#     all_splits = text_splitter.split_documents(docs)
    
#     vectorstore = Chroma.from_documents(documents=all_splits, embedding=HuggingFaceEmbeddings())
#     return vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})

# retriever = load_and_process_pdf()
# # Initialize LLM
# llm = ChatGroq(model="llama3-8b-8192")

# # Set up the RAG chain
# def format_docs(docs):
#     return "\n\n".join(doc.page_content for doc in docs)

# # Define the RAG prompt template
# template = """Use the following pieces of context to answer.
# You are directly answering to recruiter so don't say anything neagtive about Utsav Soni and his professional career.
# Read question carefully and answer based on the question asked. Your answer must be related and properly aligen to question.
# If you don't know the answer or negative question asked, just say that Thank you for bringing that up! I understand your concern, for more information contact Utsav Soni.
# Don't make up an answer and lighten up the user about excellent things from Utsav Soni's Professional Career.
# Do not reply to rephrase the question asked, if question is greeting do reply greeting only that's it.
# Use five to six sentences and keep the answer as concise, positive, polite, professional as possible.
# If user or ecrruiter ask negative qustion about Utsav Soni, please answer that in postive way by embracing Utsav Soni's skill according to question asked.
# Never ever say negative thing and negative thought about Utsav Soni in your completions.
# Utsav Soni is very good candidate to hire kee in mind. It's very important to hire him. So give smart answers only.
# You are not supposed to rate Utsav Soni for any of his skill or experience for any job role.
# You are not supposed to give reasons and facts why Utsav Soni is not selected for next round or why he is not best fit for any of the post recruiter will mention.
# If you find the question completely irrelevant like question is not asking something about Utsav Soni or his resume, please reply I am not trained for this. Thanks!
# Always say "Thanks for asking..!" at the end of the answer.

# {context}

# Question: {question}

# Smart Answer:"""
# custom_rag_prompt = PromptTemplate.from_template(template)



# rag_chain = (
#     {"context": retriever | format_docs, "question": RunnablePassthrough()}
#     | custom_rag_prompt
#     | llm
#     | StrOutputParser()
# )

# # Streamlit app
# def main():
#     st.title("Utsav Soni Resume Q&A :::")

#     user_question = st.text_input("Ask a question about Utsav Soni's resume:")

#     if user_question:
#         with st.spinner("Generating answer..."):
#             response = rag_chain.invoke(user_question)
#         st.write(response)

# if __name__ == "__main__":
#     main()

