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

decompose_prompt =  PromptTemplate(
    input_variables=["schema", "query"],
    template=
    """
    Given the schema: {schema}, and the query: {query}, decompose the query into a series of relevant subqueries. 
    Each subquery will be used to invoke a GraphCypherQAChain to retrieve the answer.
    Examples:
    1. If the query asks you to give information on the risks associated with a particular gene, give information from all nodes, relations or properties that are involved with risk for instance risk score, SMR, colocalization, sources etc.
    2. If the query asks for a Gene Ontology, give all the relevant information on the gene ontology
    3. If statistical significance is mentioned, ensure that p-value is < 0.05.
    4. If asked about directionality, look at the beta value
    5. Properties such as p-value, beta, risk sources etc. are stores as relation properties and not node properties. So, ensure that you are looking at the correct properties.
    6. For questions that have a simple answer, such as "What is the name of the gene?" or "What does this annotation term mean?" Just output 1 subquery that retrieves the answer.
    7. Do not generate redundant subqueries such as outputting the gene node when a simple count is asked for.
    
    If a query can be answered by a single subquery, DO NOT output multiple subqueries.

    KEEP THE NUMBER OF SUBQUERIES TO A MINIMUM. 
    DO NOT MAKE SUBQUERIES THAT GIVE SAME INFORMATION.
    DO NOT MAKE SUBQUERIES THAT ARE NOT RELEVANT TO THE QUERY.
    DO NOT OUTPUT A CYPHER QUERY AS A SUBQUERY.
    DO NOT MAKE DUPLICATE SUBQUERIES.
    MAXIMUM OF 3 SUBQUERIES.

    Output the subqueries in a numbered list, with each subquery on a new line.
    """
)

decompose_chain = LLMChain(llm=llm4, prompt=decompose_prompt)

def decompose_query(query: str, schema: str) -> List[str]:
    result = decompose_chain.run(schema=schema, query=query)
    # Split the result into individual subqueries
    subqueries = [sq.strip() for sq in result.split('\n') if sq.strip() and sq[0].isdigit()]
    return subqueries

# Defining a prompt template
CYPHER_GENERATION_TEMPLATE = """
You are an expert Neo4j Developer translating user questions into Cypher to answer questions about bioinformatics and biology from given Knowledge Graphs

Instructions:
ONLY ANSWER IN CYPHER QUERIES
RETURN CORRECT QUERIES ONLY
P-VALUE AND BETA ARE ALWAYS RELATIONSHIP PROPERTIES CALLED AS:
(g:GENE)-[r:HAS_GWAS_RISK]->(gwas:RISK_GWAS) RETURN r.p_value, r.beta (EXAMPLE JUST FOR GWAS)

Schema: {schema}
Question: {question}
"""

# Initializing prompt Template
cypher_generation_prompt = PromptTemplate(
    template=CYPHER_GENERATION_TEMPLATE,
    input_variables=["schema", "question"],
)

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
compile_prompt =  PromptTemplate(
    input_variables=["query", "results"],
    template=
    """
    You are a domain expert in bioinformatics and biology and will compile cypher query results into a single coherent answer given the original query:{query} And the following results from subqueries: {results}
    DO NOT cite anything
    DO NOT include the original query in the answer
    CAN use existing knowledge to compile the answer
    If information is missing, answer based on your existing knowledge
    PROVIDE DISCLAIMER if using existing knowledge
    Be concise and to the point
    """
)

compile_chain = LLMChain(llm=llm4, prompt=compile_prompt)

def compile_results(query: str, results: List[str]) -> str:
    compiled_result = compile_chain.run(query=query, results="\n\n".join(results))
    return compiled_result

def process_query(query: str) -> str:
    # Get the schema
    schema = graph.schema.split("\n")
    
    # Decompose query
    subqueries = decompose_query(query, schema)
    
    # Print subqueries
    print("Generated subqueries:")
    for i, subquery in enumerate(subqueries, 1):
        print(f"{subquery}")
    print("\n")
    
    # Execute subqueries
    results = []
    for subquery in subqueries:
        result = cypher_chain.invoke({"query": subquery})
        results.append(f"Subquery: {subquery}\nResult: {result['result']}")
    
    # Compile results
    final_result = compile_results(query, results)
    
    return final_result

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

if prompt := st.chat_input("Ask a question about bioinformatics"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Add a checkbox for additional functionality
    stringdb_checkbox = st.checkbox("Include additional functionality with STRING DB API")
    
    if stringdb_checkbox:
        genes = ["JAKMIP1", "CAPN7", "FCGR2A", "UBA3", "ATF6", "AGPAT1", "LTB", "CALML4", "IQCA1L", "RIPK2", "RASA2", "TIAM1", "CD6", "TFRC", "CD8A", "ERN1", "INPP5D", "NEDD4"]
        stringdb_url = get_stringdb_info(genes)
        full_response = process_query(prompt) + f"\n\nSTRING DB Network: {stringdb_url}"
    else:
        full_response = process_query(prompt)
    
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown(full_response)
    st.session_state.messages.append({"role": "assistant", "content": full_response})
