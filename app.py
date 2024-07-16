import streamlit as st
import toml
from langchain_openai import ChatOpenAI
from langchain_community.graphs import Neo4jGraph
from langchain.chains import GraphCypherQAChain
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from typing import List
import requests

# Load secrets
secrets = toml.load("streamlit/secrets.toml")

st.title("Bioinformatics Chatbot")

# OpenAI API Key
llm4 = ChatOpenAI(openai_api_key=st.secrets["OPENAI_API_KEY"], model="gpt-4o", temperature=0)
llm3_5 = ChatOpenAI(openai_api_key=st.secrets["OPENAI_API_KEY"], model="gpt-3.5-turbo", temperature=0)

# Neo4j Graph connection
graph = Neo4jGraph(
    url=st.secrets["url"],
    username=st.secrets["username"],
    password=st.secrets["password"]
)

decompose_and_generate_prompt =  PromptTemplate(
    input_variables=["schema", "query"],
    template=
    """
    Given the schema: {schema}, and the query: {query}, decompose the query into relevant subqueries and generate corresponding Cypher queries.
    Output the Cypher queries in a numbered list, with each query on a new line.

    Example:
    1. MATCH (go:GENE_ONTOLOGY {annotation_term: "Cytoplasm"}) RETURN go
    2. MATCH (p:Protein)-[:ANNOTATED_WITH]->(go:GENE_ONTOLOGY {annotation_term: "Cytoplasm"}) RETURN p
    3. MATCH (g:GENE)-[:ENCODES]->(p:Protein)-[:ANNOTATED_WITH]->(go:GENE_ONTOLOGY {annotation_term: "Cytoplasm"}) RETURN g
    """
)

decompose_and_generate_chain = LLMChain(llm=llm4, prompt=decompose_and_generate_prompt)

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

def decompose_and_generate_queries(query: str, schema: str) -> List[str]:
    result = decompose_and_generate_chain.run(schema=schema, query=query)
    # Split the result into individual Cypher queries
    cypher_queries = [cq.strip() for cq in result.split('\n') if cq.strip() and cq[0].isdigit()]
    return cypher_queries

def compile_results_and_extract_genes(query: str, results: List[str]) -> (str, List[str]):
    result = compile_and_extract_chain.run(query=query, results="\n\n".join(results))
    compiled_answer, genes_text = result.split("\n\n")
    gene_symbols = [gene.strip() for gene in genes_text.split(',')]
    return compiled_answer, gene_symbols

def process_query(query: str, include_stringdb: bool) -> str:
    # Get the schema
    schema = graph.schema.split("\n")
    
    # Decompose query and generate Cypher queries in a single call
    cypher_queries = decompose_and_generate_queries(query, schema)
    
    # Print Cypher queries
    print("Generated Cypher queries:")
    for i, cypher_query in enumerate(cypher_queries, 1):
        print(f"{cypher_query}")
    print("\n")
    
    # Execute Cypher queries
    results = []
    for cypher_query in cypher_queries:
        result = cypher_chain.invoke({"query": cypher_query})
        results.append(f"Cypher query: {cypher_query}\nResult: {result['result']}")
    
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

# Initialize the chain
cypher_chain = GraphCypherQAChain.from_llm(
    llm4,
    graph=graph,
    cypher_prompt=cypher_generation_prompt,
    verbose=True,
    return_intermediate_steps=False,
    return_intermediate_results=False,
    top_k = 50
)

if prompt:
    full_response = process_query(prompt, stringdb_checkbox)
    
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown(full_response)
    st.session_state.messages.append({"role": "assistant", "content": full_response})
