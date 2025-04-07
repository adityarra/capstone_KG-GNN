from langchain_neo4j import Neo4jGraph
import os
from langchain.vectorstores.neo4j_vector import Neo4jVector
from langchain_community.embeddings import OpenAIEmbeddings


os.environ['OPENAI_API_KEY'] = "OPENAI_API_KEY"

graph = Neo4jGraph(
    url="neo4j+s://a2b2250f.databases.neo4j.io",
    username="neo4j",
    password="5o0ApLLdTw4z79C4thgPR1AIeuppSIrJgpIgOjkobPk"
)

response = graph.similarity_search(
    "How will RecommendationService be updated?"
)
print(response[0].page_content)
