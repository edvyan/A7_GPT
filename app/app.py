import dash
from dash import html, dcc, Input, Output
from langchain.chains import LLMChain
from langchain.chains.conversational_retrieval.prompts import CONDENSE_QUESTION_PROMPT
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import ConversationalRetrievalChain
from transformers import AutoTokenizer, pipeline, AutoModelForSeq2SeqLM
from langchain import HuggingFacePipeline
import torch
from langchain import PromptTemplate
from langchain_community.vectorstores import FAISS
import os
from langchain_community.embeddings import HuggingFaceInstructEmbeddings

app = dash.Dash(__name__)

model_name = 'hkunlp/instructor-base'

embedding_model = HuggingFaceInstructEmbeddings(
    model_name = model_name,
    model_kwargs = {"device" : 'cpu'}
)


# Take vector
vector_path = '../vector-store'
db_file_name = 'nlp_stanford'

from langchain.vectorstores import FAISS

vectordb = FAISS.load_local(
    folder_path = os.path.join(vector_path, db_file_name),
    embeddings = embedding_model,
    index_name = 'nlp' #default index
)

retriever = vectordb.as_retriever()


prompt_template = """
    I'm your GTP bot named AIT GPT.
    I will assist you around to let you know more about AIT.
    Please feel free to ask any questions regarding AIT.
    {context}
    Question: {question}
    Answer:
    """.strip()

PROMPT = PromptTemplate.from_template(
    template = prompt_template
)

model_id = '../models/fastchat-t5-3b-v1.0/'

tokenizer = AutoTokenizer.from_pretrained(
    model_id)

tokenizer.pad_token_id = tokenizer.eos_token_id

model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

pipe = pipeline(
    task="text2text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens = 256,
    model_kwargs = {
        "temperature" : 0,
        "repetition_penalty": 1.5
    }
)

llm = HuggingFacePipeline(pipeline = pipe)

question_generator = LLMChain(
    llm = llm,
    prompt = CONDENSE_QUESTION_PROMPT,
    verbose = True
)

doc_chain = load_qa_chain(
    llm = llm,
    chain_type = 'stuff',
    prompt = PROMPT,
    verbose = True
)

memory = ConversationBufferWindowMemory(
    k=3, 
    memory_key = "chat_history",
    return_messages = True,
    output_key = 'answer'
)

chain = ConversationalRetrievalChain(
    retriever=retriever,
    question_generator=question_generator,
    combine_docs_chain=doc_chain,
    return_source_documents=True,
    memory=memory,
    verbose=True,
    get_chat_history=lambda h : h
)

# Layout
app.layout = html.Div([
    dcc.Input(id='input-query', type='text', placeholder='Enter your question'),
    html.Button('Submit', id='submit-query'),
    html.Div(id='output-answer'),
    html.Div(id='output-source')
])

# Callbacks for interactivity
@app.callback(
    [Output('output-answer', 'children'),
     Output('output-source', 'children')],
    [Input('submit-query', 'n_clicks')],
    [dash.dependencies.State('input-query', 'value')]
)
def update_output(n_clicks, value):
    if n_clicks is None:
        return '', ''
    else:
        answer = chain({"question": value})['answer']
        source = chain({"question": value})['source_documents']
        return answer, source

# Run the Dash app
if __name__ == '__main__':
    app.run_server(debug=True)