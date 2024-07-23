import streamlit as st
import redis
import hashlib
from langchain_openai import ChatOpenAI
from langchain_community.graphs import Neo4jGraph
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from neo4j import GraphDatabase
from typing import List
import time

st.set_page_config(
    page_title="Bioinformatics Chatbot",
    page_icon="n23_icon.png",
    initial_sidebar_state="expanded",
    layout="wide"
)

# Key variables
api_key = st.secrets["OPENAI_API_KEY"]
URL = st.secrets["url"]
AUTH = (st.secrets["username"], st.secrets["password"])
driver = GraphDatabase.driver(URL, auth=AUTH)
driver.verify_connectivity()
graph = Neo4jGraph(
    url=st.secrets["url"],
    username=st.secrets["username"],
    password=st.secrets["password"]
)

# Redis setup
redis_client = redis.StrictRedis(
    host=st.secrets["redis_host"],
    port=12019,
    password=st.secrets["redis_password"],
    db=0,
    decode_responses=True
)


# Function to create a unique key for caching
def create_cache_key(query: str, schema: str):
    return hashlib.sha256((query + schema).encode()).hexdigest()

# Function to cache and retrieve results from Redis
def cache_results(key: str, results=None):
    if results:
        redis_client.set(key, str(results))
        return results
    cached_results = redis_client.get(key)
    if cached_results:
        return eval(cached_results)
    return None

# Function to store conversation context
def store_context(session_id: str, question: str, answer: str):
    context_key = f"context:{session_id}"
    context_data = redis_client.get(context_key)
    if context_data:
        context = context_data
        context += f"\nUser: {question}\nBot: {answer}"
    else:
        context = f"User: {question}\nBot: {answer}"
    redis_client.set(context_key, context, ex=86400)  # Set TTL to 24 hours

# Function to retrieve conversation context
def get_context(session_id: str) -> str:
    context_key = f"context:{session_id}"
    context_data = redis_client.get(context_key)
    if context_data:
        return context_data
    return ""

# OpenAI API initialization
def get_openai_llm(api_key, model, temperature, max_tokens):
    return ChatOpenAI(openai_api_key=api_key, model=model, temperature=temperature, max_tokens=max_tokens)


# Define the prompt templates
cypher_generation_prompt = PromptTemplate(
    template="""
    You are an expert Neo4j Developer translating user questions into Cypher to answer questions about bioinformatics and biology from given Knowledge Graphs.

    Instructions:
    ONLY RETURN THE CYPHER QUERY
    DO NOT INCLUDE ANY EXPLANATION, NATURAL LANGUAGE TEXT, OR CODE BLOCKS
    ALWAYS CALL RELATIONSHIPS r
    FOR INTERACTIONS CHECK FOR BOTH DIRECTIONS
    P-VALUE AND BETA ARE ALWAYS RELATIONSHIP PROPERTIES CALLED AS:
    (g:GENE)-[r:HAS_GWAS_RISK]->(gwas:RISK_GWAS) RETURN r.p_value, r.beta (EXAMPLE JUST FOR GWAS)

    Schema: {schema}
    Question: {question}
    """,
    input_variables=["schema", "question"],
)

compile_prompt = PromptTemplate(
    input_variables=["query", "results"],
    template="""
    You are a domain expert in bioinformatics and biology and will compile cypher query results into a single coherent answer given the original query: {query} and the following results from subqueries: {results}.
    DO NOT cite anything.
    DO NOT include the original query in the answer.
    CAN use existing knowledge to compile the answer.
    If information is missing, answer based on your existing knowledge.
    PROVIDE DISCLAIMER if using existing knowledge.
    Be concise and to the point.
    Format the response using Markdown:
    - Use **bold** for important terms
    - Use bullet points for lists
    """
)

# Streamlit app
st.set_page_config(page_title="Bioinformatics Knowledge Graph Chatbot", initial_sidebar_state="expanded", layout="wide")

# Title and sidebar inputs
st.title("Bioinformatics Knowledge Graph Chatbot")

with st.sidebar:
    st.markdown("# Chat Options")
    model = st.selectbox("Select GPT Model", ["gpt-4o-mini", "gpt-4o"])
    max_tokens = st.number_input("Output Token Length", min_value=1, max_value=4096, value=4096)
    temperature = st.slider("Temperature", min_value=0.0, max_value=0.5, value=0.01)


# Initialize LLMs
llm4 = get_openai_llm(api_key, model, temperature, max_tokens)
cypher_chain = LLMChain(llm=llm4, prompt=cypher_generation_prompt)
compile_chain = LLMChain(llm=llm4, prompt=compile_prompt)
schema = graph.schema

# Function to generate and execute Cypher queries
def process_query(session_id: str, query: str, schema: str):
    cache_key = create_cache_key(query, schema)
    cached_results = cache_results(cache_key)
    if cached_results:
        results = cached_results
    else:
        cypher_query = cypher_chain.run({"schema": schema, "question": query})
        cypher_query = cypher_query.strip().replace("```cypher", "").replace("```", "").strip()

        with driver.session() as session:
            results = session.run(cypher_query)
            results = [record.data() for record in results]

        cache_results(cache_key, results)

    final_answer_placeholder = st.empty()
    
    def generate_final_answer():
        partial_answer = ""
        final_response = compile_chain.run({"query": query, "results": results})
        for char in final_response:
            partial_answer += char
            final_answer_placeholder.markdown(partial_answer)
            time.sleep(0.01)
        return final_response
    
    final_response = generate_final_answer()

    st.subheader("Generated Cypher Query")
    st.code(cypher_query)
    
    return final_response

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

session_id = st.session_state.get('session_id', str(time.time()))
st.session_state['session_id'] = session_id

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask a question about bioinformatics"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        context = get_context(session_id)
        full_response = process_query(session_id, f"{context}\n{prompt}", schema)
        st.session_state.messages.append({"role": "assistant", "content": full_response})
        store_context(session_id, prompt, full_response)
