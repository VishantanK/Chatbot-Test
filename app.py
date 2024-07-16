import streamlit as st
import toml
from langchain_openai import ChatOpenAI
from langchain_community.graphs import Neo4jGraph
from langchain.chains import GraphCypherQAChain
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from typing import List, Tuple
import requests

# Load secrets
secrets = toml.load("streamlit/secrets.toml")

st.title("Bioinformatics Chatbot")

# OpenAI API Key
llm4 = ChatOpenAI(openai_api_key=st.secrets["OPENAI_API_KEY"], model="gpt-4o", temperature=0)

# Neo4j Graph connection
graph = Neo4jGraph(
    url=st.secrets["url"],
    username=st.secrets["username"],
    password=st.secrets["password"]
)

compile_and_extract_prompt =  PromptTemplate(
    input_variables=["query", "results"],
    template=
    """
    You are a domain expert in bioinformatics and biology. Compile the results from the Cypher queries into a coherent answer for the original query. 
    Also, extract all gene symbols from the final answer. Gene symbols are typically all uppercase and may include numbers. 
    Return the compiled answer followed by the gene symbols as a comma-separated list.

    Original query: {query}
    Cypher query results: {results}
    """
)

compile_and_extract_chain = LLMChain(llm=llm4, prompt=compile_and_extract_prompt)

def compile_results_and_extract_genes(query: str, results: List[str]) -> Tuple[str, List[str]]:
    inputs = {
        "query": query,
        "results": "\n\n".join(results)
    }
    result = compile_and_extract_chain.run(inputs)
    compiled_answer, genes_text = result.split("\n\n")
    gene_symbols = [gene.strip() for gene in genes_text.split(',')]
    return compiled_answer, gene_symbols

def process_query(query: str, include_stringdb: bool) -> str:
    # Use GraphCypherQAChain to decompose, generate Cypher, and execute in one step
    qa_chain = GraphCypherQAChain.from_llm(
        llm=llm4,
        graph=graph,
        verbose=True,
        return_intermediate_steps=False,
        return_intermediate_results=False,
        top_k=50
    )

    result = qa_chain.run(query=query)
    results = [f"Result: {result}"]

    # Compile results and extract gene symbols in a single call
    final_answer, gene_symbols = compile_results_and_extract_genes(query, results)
    
    if include_stringdb and gene_symbols:
        stringdb_url = get_stringdb_info(gene_symbols)
        final_answer += f"\n\nSTRING DB Network: {stringdb_url}"
    
    return final_answer

def get_stringdb_info(genes: List[str]) -> str:
    base_url = "https://string-db.org/cgi/network?identifiers="
    gene_string = "%0d".join(genes)
    url = f"{base_url}{gene_string}&species=9606&show_query_node_labels=1"
    response = requests.get(url)
    if response.status_code == 200:
        return url
    else:
        return "Failed to retrieve data from STRING DB"

# Streamlit chat interface
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Use Streamlit columns to place the input and checkbox side by side
cols = st.columns([4, 1])
with cols[0]:
    if prompt := st.chat_input("Ask a question about bioinformatics"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

with cols[1]:
    stringdb_checkbox = st.checkbox("Include STRING DB")

if prompt:
    full_response = process_query(prompt, stringdb_checkbox)
    
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown(full_response)
    st.session_state.messages.append({"role": "assistant", "content": full_response})
