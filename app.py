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
    redis_client.set(context_key, context, ex=1200)  # Set TTL to 24 hours

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
    - ONLY RETURN THE CYPHER QUERY
    - DO NOT INCLUDE ANY EXPLANATION, NATURAL LANGUAGE TEXT, OR CODE BLOCKS
    - FOR DRUGGABILITY CHECK IF IT IS TRUE OR FALSE
    - MAKE COMPLICATED OR MULTISTEP QUERIES IF NEEDED
    - IF ASKED ABOUT RISK SOURCES OR HOW RISK SCORE IS CALCULATED CHECK THE 'Source' RELATIONSHIP PROPERTY 
        - EXAMPLE : "MATCH (g:Gene)-[r:HAS_RISK_SCORE]->(rs:Risk_Score) RETURN rs.score, r.Source"
    - IF SIGNIFICANCE OR STATISTICAL SIGNIFICANCE FOR GWAS OR SMR CHECK p-value < 0.05 FROM THE RELATIONSHIP PROPERTIES
    
    Schema: {schema}
    Question: {question}
    """,
    input_variables=["schema", "question"],
)

kg_generation_prompt = PromptTemplate(
    template="""
    You are an expert Neo4j Developer creating a Cypher query to generate a knowledge graph relevant to the given question.

    Instructions:
    ONLY RETURN THE CYPHER QUERY
    DO NOT INCLUDE ANY EXPLANATION, NATURAL LANGUAGE TEXT, OR CODE BLOCKS
    MAKE SURE THE QUERY IS RELEVANT TO THE QUESTION ASKED
    RETURN ALL NODES RELEVANT TO THE QUESTION
    RETURN ALL RELATIONSHIPS RELEVANT TO THE QUESTION
    DO NOT RETURN PROPERTIES OF NODES OR RELATIONSHIPS

    Schema: {schema}
    Question: {question}
    """,
    input_variables=["schema", "question"],
)

compile_prompt = PromptTemplate(
    input_variables=["query", "results"],
    template="""
    - You are a domain expert in bioinformatics and biology and will compile cypher query results into a single coherent answer given the original query: {query} and the following results from subqueries: {results}.
    - DO NOT cite anything.
    - DO NOT include the original query in the answer.
    - CAN use existing knowledge to compile the answer.
    - If information is missing, answer based on your existing knowledge.
    - PROVIDE DISCLAIMER if using existing knowledge.
    - Be concise and to the point.
    - Format the response using Markdown:
    - Use **bold** for important terms
    - Use bullet points for lists
    - Tables are allowed
    - Risk score sources can be as follows: Specify the source properly when asked
        - GWAS hits
        - SMR : Can be either pQTL or eQTL sourced from different tissues and Data sources - GTex, Metabrain, Yang, UKB
        - Colocalization : Can be either pQTL or eQTL sourced from different tissues and Data sources Data sources - GTex, Metabrain, Yang, UKB
    - DO NOT REPEAT THE ANSWER FROM THE CONTEXT
    
    """
)

# Title and sidebar inputs
st.title("Bioinformatics Knowledge Graph Chatbot")

with st.sidebar:
    st.markdown("# Chat Options")
    model = st.selectbox("Select GPT Model", ["gpt-4o", "gpt-4o-mini"])
    max_tokens = st.number_input("Output Token Length", min_value=1, max_value=4096, value=4096)
    temperature = st.slider("Temperature", min_value=0.0, max_value=0.5, value=0.01)
    generate_kg = st.checkbox("Generate Knowledge Graph")


# Initialize LLMs
llm4 = get_openai_llm(api_key, model, temperature, max_tokens)
llm4mini = get_openai_llm(api_key, "gpt-4o-mini", temperature, max_tokens)
cypher_chain = LLMChain(llm=llm4, prompt=cypher_generation_prompt)
kg_chain = LLMChain(llm=llm4, prompt=kg_generation_prompt)
compile_chain = LLMChain(llm=llm4mini, prompt=compile_prompt)
schema = graph.schema

# Function to visualize the graph
def visualize_graph(results: List):
    net = Network(height="500px", width="100%", directed=True)
    for record in results:
        for node in record.data():
            net.add_node(node, label=node)
        for node in record.data():
            for rel in record.data()[node]:
                net.add_edge(node, rel, label=rel)
    net.show("graph.html")
    HtmlFile = open("graph.html", "r", encoding="utf-8")
    source_code = HtmlFile.read()
    components.html(source_code, height=600)

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
            results = list(results)  # Convert to list to preserve Neo4j types

        cache_results(cache_key, results)

    final_answer_placeholder = st.empty()
    
    def generate_final_answer():
        partial_answer = ""
        final_response = compile_chain.run({"query": query, "results": str(results)})
        for char in final_response:
            partial_answer += char
            final_answer_placeholder.markdown(partial_answer)
            time.sleep(0.01)
        return final_response
    
    final_response = generate_final_answer()

    #st.subheader("Generated Cypher Query")
    #st.code(cypher_query)

    st.subheader("Query Results")
    st.write(pd.DataFrame([record.data() for record in results]))

    if generate_kg:
        kg_query = kg_chain.run({"schema": schema, "question": query})
        kg_query = kg_query.strip().replace("```cypher", "").replace("```", "").strip()
        st.subheader("Knowledge Graph Cypher Query")
        st.code(kg_query)
        st.markdown("To visualise the KG, input the cypher query [here](https://browser.neo4j.io/?_gl=1*1mqlp1u*_gcl_aw*R0NMLjE3MjE0MTI1NzAuQ2owS0NRanctdUswQmhDMEFSSXNBTlF0Z0dPZHI2MFdBbUkxUmtFeXNfelNtdkFWOEZLajJSRWVMTjR2b1BZMGhGNUdjM3loQXo4R0wxWWFBamJyRUFMd193Y0I.*_gcl_au*ODQwMDI2NTMyLjE3MTg3Mjk5ODQ.*_ga*MTUzMjMxMzIyMS4xNzE4NzI5OTgz*_ga_DL38Q8KGQC*MTcyMTY2MTc0Ni4xMy4xLjE3MjE2NjE3NTMuMC4wLjA.*_ga_DZP8Z65KK4*MTcyMTY2MTc0Ni4yNC4xLjE3MjE2NjE3NTMuMC4wLjA.)", unsafe_allow_html=True)

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
        full_response= process_query(session_id, f"Context: {context} \n Prompt: {prompt}", schema)
        st.session_state.messages.append({"role": "assistant", "content": full_response})
        store_context(session_id, prompt, full_response)
        
