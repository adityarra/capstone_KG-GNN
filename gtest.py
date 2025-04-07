from neo4j import GraphDatabase
from langchain_neo4j import Neo4jGraph
from langchain_community.embeddings import OpenAIEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import os

os.environ['OPENAI_API_KEY'] = "AIzaSyCZAW3PhtgoQq2MzdtWTuWR3WyCnn4ZF70"
embeddings = OpenAIEmbeddings()

# Neo4j connection details
uri = "neo4j+s://demo.neo4jlabs.com"  # Using 'bolt' protocol for Neo4j
username = "movies"  # Replace with the correct username (usually 'neo4j')
password = "movies"  # Replace with your Neo4j password

# Establish a connection to the Neo4j database
driver = GraphDatabase.driver(uri, auth=(username, password))

def run_query():
    # Open a session and run a basic query
    with driver.session() as session:
        query = "MATCH (n) RETURN n"  # Basic query to fetch first 5 nodes
        result = session.run(query)

# Run the query
run_query()

# Close the connection
driver.close()

query = "What is the most used supply?"

query_embedding = embeddings.embed_query(query)

vectors = []
nodes = []
for record in result:
    node = record["n"]
    vector = record["vector"]  
    nodes.append(node)
    vectors.append(vector)

similarities = cosine_similarity([query_embedding], vectors)[0]
similar_nodes = sorted(zip(similarities, nodes), reverse=True)
print(similar_nodes[0][1])