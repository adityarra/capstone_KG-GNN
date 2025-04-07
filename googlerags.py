import torch
from langchain_neo4j import Neo4jGraph
from langchain_community.embeddings import OpenAIEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel
from neo4j import GraphDatabase
import os

# os.environ['GOOGLE_API_KEY'] = "AIzaSyDZ7RgqRSO0UN00fUtpJ7OpZhGXoFF7WgM"
# aiplatform.init(project="vernal-period-451113-k8", location="pes.edu")

model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)


# uri="neo4j://9ce782dd.databases.neo4j.io",
# username="neo4j",
# password="tv_TrxHUAwytzXBwh8p6t1VEYiA-35PYQrlmTbYnkOs"
# driver = GraphDatabase.driver(uri, auth=(username, password))

uri="neo4j+s://demo.neo4jlabs.com"
username="movies"
password="movies"
driver = GraphDatabase.driver(uri, auth=(username, password))

def get_query_embedding(query: str):
    # Tokenize the input query text
    inputs = tokenizer(query, return_tensors="pt", truncation=True, padding=True, max_length=512)
    
    # Get the embeddings from the model (we take the mean of the last layer's hidden states)
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Extract the embeddings from the model output (mean of the last hidden states)
    embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
    
    return embeddings

def similarity_search(query: str):
    query_embedding = get_query_embedding(query)
    
    query_string = """
    MATCH (n)
    RETURN n,n.embedding AS embedding
    """

    with driver.session() as session:
        result = session.run(query_string)
        
        # Collect nodes and their embeddings
        vectors = []
        nodes = []
        for record in result:
            node = record["n"]
            vector = record["embedding"]
            nodes.append(node)
            vectors.append(vector)


    similarities = cosine_similarity([query_embedding], vectors)[0]
    
    similar_nodes = sorted(zip(similarities, nodes), reverse=True)
    
    top_node = similar_nodes[0][1]  
    return top_node

response = similarity_search("What is the most used supply?")

print(response) 
