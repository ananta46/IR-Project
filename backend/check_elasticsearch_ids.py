from elasticsearch import Elasticsearch
import pandas as pd

# Elasticsearch connection (use the same settings as in your server.py)
es_client = Elasticsearch(
    "https://localhost:9200", 
    basic_auth=("elastic", "+zu*TMbwCT-I9_fi3-L4"), 
    ca_certs="~/http_ca.crt",
    verify_certs=True
)

# Index name
INDEX_NAME = "recipes"

# Check if index exists
if not es_client.indices.exists(index=INDEX_NAME):
    print(f"ERROR: Index {INDEX_NAME} does not exist")
    exit()

# Get a sample of documents
search_body = {
    "size": 5,
    "query": {"match_all": {}}
}

response = es_client.search(index=INDEX_NAME, body=search_body)
hits = response["hits"]["hits"]

print(f"Total recipes in Elasticsearch: {response['hits']['total']['value']}")
print("\nSample documents from Elasticsearch:")

for hit in hits:
    source = hit["_source"]
    print(f"Recipe ID: {source.get('RecipeId', 'Not found')}, Type: {type(source.get('RecipeId', ''))}")
    print(f"Recipe Name: {source.get('Name', 'Unnamed')}")
    print("---")

# Check for specific IDs
print("\nChecking for specific IDs in Elasticsearch:")

ids_to_check = ["38551", "33851", "29782", "30578", "47276", 
                "12345", "56789", "44732", "39725", "20619"]

for recipe_id in ids_to_check:
    search_body = {
        "query": {
            "term": {
                "RecipeId": recipe_id
            }
        }
    }
    
    response = es_client.search(index=INDEX_NAME, body=search_body)
    hits = response["hits"]["hits"]
    
    if hits:
        print(f"Recipe ID {recipe_id} EXISTS in Elasticsearch")
    else:
        print(f"Recipe ID {recipe_id} NOT FOUND in Elasticsearch")