from langchain_neo4j import Neo4jGraph
from langchain_community.embeddings import OpenAIEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import os

os.environ['OPENAI_API_KEY'] = "AIzaSyCZAW3PhtgoQq2MzdtWTuWR3WyCnn4ZF70"

graph = Neo4jGraph(
    url="neo4j+s://demo.neo4jlabs.com",
    username="wordnet",
    password="wordnet"
)

embeddings = OpenAIEmbeddings()

def get_query_embedding(query: str):
    return embeddings.embed_query(query)

def similarity_search(query: str):
    query_embedding = get_query_embedding(query)
    
    query_string = """
    MATCH (n)
    RETURN n
    """
    
    result = graph.run(query_string)

    vectors = []
    nodes = []
    for record in result:
        node = record["n"]
        vector = record["vector"]  
        nodes.append(node)
        vectors.append(vector)

    similarities = cosine_similarity([query_embedding], vectors)[0]
    
    similar_nodes = sorted(zip(similarities, nodes), reverse=True)
    
    top_node = similar_nodes[0][1]  
    return top_node

response = similarity_search("What is the most used supply?")

print(response) 
