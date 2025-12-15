import cohere 
from qdrant_client import QdrantClient

cohere_client = cohere.Client('2QrWB3DWqq0uCCvhTAgGUTw0u1IV2W2REjP3Q4d0')

qdrant = QdrantClient(
    url="https://415c1e57-4c19-4951-acd7-3336aa42b24f.europe-west3-0.gcp.cloud.qdrant.io:6333",
    api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.bMcgrK4L01ED9tmN6KMVgAflJSGqhyhraQtMYlaaHMA",

)


def get_embedding(text):
    """Get embedding vector from Cohere Embed v3"""
    response = cohere_client.embed(
        model="embed-english-v3.0",
        input_type="search_query",  # Use search_query for queries
        texts=[text],
    )
    return response.embeddings[0]  # Return the first embedding


def retrieve(query):
    embedding = get_embedding(query)
    result = qdrant.query_points(
        collection_name="humanoid_ai_book",
        query=embedding,
        limit=5
    )
    return [point.payload["text"] for point in result.points]

# Test
print(retrieve("What data do you have?"))