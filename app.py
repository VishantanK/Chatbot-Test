import streamlit as st
import toml
from langchain_openai import ChatOpenAI
from langchain_community.graphs import Neo4jGraph
from langchain.chains import GraphCypherQAChain
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from typing import List
import requests
import pyshorteners


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

# Improved decomposition prompt template
decompose_prompt = PromptTemplate(
    input_variables=["schema", "query"],
    template="""
    Given the schema: {schema} and the query: {query}, decompose the query into the minimum number of necessary subqueries to retrieve the relevant information. Ensure that each subquery is unique and relevant. Avoid redundancy and irrelevant subqueries.

    Examples:
    1. If the query is about the most significant GWAS hit for a gene, provide only the relevant information like GWAS hit details (id, chr, pos, ea, oa, beta, pval) and associated risk.
    2. If statistical significance is mentioned, ensure the p-value is < 0.05.
    3. If asked about directionality, look at the beta value.

    Output the subqueries in a numbered list, with each subquery on a new line.
    """
)

decompose_chain = LLMChain(llm=llm4, prompt=decompose_prompt)

def decompose_query(query: str, schema: str) -> List[str]:
    result = decompose_chain.run(schema=schema, query=query)
    subqueries = [sq.strip() for sq in result.split('\n') if sq.strip() and sq[0].isdigit()]
    return subqueries

# Improved Cypher query generation template
cypher_generation_prompt = PromptTemplate(
    template="""
    You are an expert Neo4j Developer translating user questions into Cypher to answer questions about bioinformatics and biology from given Knowledge Graphs.

    Instructions:
    ONLY ANSWER IN CYPHER QUERIES.
    RETURN CORRECT QUERIES ONLY.
    P-VALUE AND BETA ARE ALWAYS RELATIONSHIP PROPERTIES.

    Schema: {schema}
    Question: {question}

    Example: If the question is to find the top 5 genes with the highest risk score, ensure the query sorts by the risk score and limits the results to the top 5.
    """,
    input_variables=["schema", "question"],
)

cypher_chain = GraphCypherQAChain.from_llm(
    llm4,
    graph=graph,
    cypher_prompt=cypher_generation_prompt,
    verbose=True,
    return_intermediate_steps=False,
    return_intermediate_results=False,
    top_k=30
)

compile_prompt = PromptTemplate(
    input_variables=["query", "results"],
    template="""
    You are a domain expert in bioinformatics and biology and will compile Cypher query results into a single coherent answer.

    Query: {query}
    Results: {results}

    DO NOT cite anything.
    PROVIDE DISCLAIMER if using existing knowledge.
    Be concise and to the point.
    """
)

compile_chain = LLMChain(llm=llm4, prompt=compile_prompt)

gene_extraction_prompt = PromptTemplate(
    input_variables=["text"],
    template="""
    Extract all gene symbols from the following text. Gene symbols are typically all uppercase and may include numbers. Return the gene symbols as a comma-separated list.

    Text: {text}
    """
)

gene_extraction_chain = LLMChain(llm=llm4, prompt=gene_extraction_prompt)

def compile_results(query: str, results: List[str], include_stringdb: bool) -> str:
    compiled_result = compile_chain.run(query=query, results="\n\n".join(results))
    
    if include_stringdb:
        # Extract gene symbols from the final response
        gene_symbols_text = gene_extraction_chain.run(text=compiled_result)
        gene_symbols = [gene.strip() for gene in gene_symbols_text.split(',')]
        if gene_symbols:
            stringdb_url = get_stringdb_info(gene_symbols)
            compiled_result += f"\n\nSTRING DB Network: {stringdb_url}"
    
    return compiled_result

def process_query(query: str, include_stringdb: bool) -> str:
    schema = graph.schema
    subqueries = decompose_query(query, schema)
    
    results = []
    for subquery in subqueries:
        try:
            result = cypher_chain.invoke({"query": subquery})
            if not result['result']:
                # If the result is empty, handle it appropriately
                results.append(f"Subquery: {subquery}\nResult: No data found in the graph database for this subquery.")
            else:
                results.append(f"Subquery: {subquery}\nResult: {result['result']}")
        except Exception as e:
            results.append(f"Subquery: {subquery}\nError: {e}")
    
    # Compile results and add disclaimer for external knowledge
    final_result = compile_results(query, results, include_stringdb)
    if "No data found" in final_result:
        additional_info = llm4.run(f"Provide information on {query} from general knowledge.")
        final_result += f"\n\n{additional_info}\n\nDisclaimer: Some information is based on existing knowledge in the field of bioinformatics and biology."
    
    return final_result

def get_stringdb_info(genes: List[str]) -> str:
    base_url = "https://string-db.org/cgi/network?identifiers="
    gene_string = "%0d".join(genes)
    url = f"{base_url}{gene_string}&species=9606&show_query_node_labels=1"
    
    type_tiny = pyshorteners.Shortener()
    
    response = requests.get(url)
    if response.status_code == 200:
        return type_tiny.tinyurl.short(url)
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
    with st.spinner('Processing your query...'):
        try:
            full_response = process_query(prompt, stringdb_checkbox)
        except Exception as e:
            full_response = f"An error occurred while processing your query: {e}"

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            message_placeholder.markdown(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})
