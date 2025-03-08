from flask import Flask, request, jsonify
from elasticsearch import Elasticsearch
import pandas as pd
import pickle
import os
import time

app = Flask(__name__)

# Elasticsearch connection
app.es_client = Elasticsearch(
    "https://localhost:9200", 
    basic_auth=("elastic", "+zu*TMbwCT-I9_fi3-L4"), 
    ca_certs="~/http_ca.crt",
    verify_certs=True
)

# Index name
INDEX_NAME = "recipes"
PICKLE_PATH = "resource/pickles/recipes_index.pkl"

def create_index():
    """Create Elasticsearch index with BM25 settings if it doesn't exist"""
    # Check if index exists
    if not app.es_client.indices.exists(index=INDEX_NAME):
        # Define BM25 settings
        settings = {
            "settings": {
                "number_of_shards": 1,
                "number_of_replicas": 0,
                "similarity": {
                    "default": {
                        "type": "BM25",
                        "b": 0.75,
                        "k1": 1.2
                    }
                },
                "analysis": {
                    "analyzer": {
                        "recipe_analyzer": {
                            "type": "custom",
                            "tokenizer": "standard",
                            "filter": ["lowercase", "stop", "snowball"]
                        }
                    }
                }
            },
            "mappings": {
                "properties": {
                    "RecipeId": {"type": "keyword"},
                    "Name": {
                        "type": "text",
                        "analyzer": "recipe_analyzer",
                        "fields": {
                            "keyword": {"type": "keyword"}
                        }
                    },
                    "Description": {"type": "text", "analyzer": "recipe_analyzer"},
                    "RecipeCategory": {"type": "keyword"},
                    "Keywords": {"type": "text", "analyzer": "recipe_analyzer"},
                    "RecipeIngredientParts": {"type": "text", "analyzer": "recipe_analyzer"},
                    "RecipeInstructions": {"type": "text", "analyzer": "recipe_analyzer"},
                    "AggregatedRating": {"type": "float"},
                    "ReviewCount": {"type": "float"},
                    "Images": {"type": "keyword"},
                    "text": {"type": "text", "analyzer": "recipe_analyzer"}
                }
            }
        }
        
        # Create the index
        app.es_client.indices.create(index=INDEX_NAME, body=settings)
        print(f"Created index {INDEX_NAME} with BM25 similarity")
    else:
        print(f"Index {INDEX_NAME} already exists")

def load_data_from_parquet():
    """Load data from Parquet file and index it in Elasticsearch in chunks"""
    parquet_path = 'resource/csv/completed_recipes.parquet'
    
    if not os.path.exists(parquet_path):
        print(f"Parquet file not found at {parquet_path}")
        return False
    
    # Load the parquet file
    df = pd.read_parquet(parquet_path)
    print(f"Loaded {len(df)} records from parquet file")
    
    # Convert DataFrame to list of dicts for bulk indexing
    records = df.to_dict(orient='records')
    
    # Process in chunks to avoid request size limits
    chunk_size = 1000  # Process 1000 records at a time
    total_chunks = (len(records) + chunk_size - 1) // chunk_size  # Ceiling division
    
    successful_chunks = 0
    
    for i in range(0, len(records), chunk_size):
        chunk = records[i:i+chunk_size]
        chunk_num = i // chunk_size + 1
        
        # Prepare bulk indexing operation for this chunk
        bulk_data = []
        for record in chunk:
            # Clean up NaN values
            cleaned_record = {k: ('' if pd.isna(v) else v) for k, v in record.items()}
            
            # Add index operation and document
            bulk_data.append({"index": {"_index": INDEX_NAME}})
            bulk_data.append(cleaned_record)
        
        # Perform bulk indexing for this chunk
        try:
            if bulk_data:
                app.es_client.bulk(body=bulk_data, refresh=(chunk_num == total_chunks))
                successful_chunks += 1
                print(f"Indexed chunk {chunk_num}/{total_chunks} ({len(chunk)} documents)")
        except Exception as e:
            print(f"Error indexing chunk {chunk_num}: {str(e)}")
    
    print(f"Completed indexing: {successful_chunks}/{total_chunks} chunks successful")
    return successful_chunks > 0

def pickle_index():
    """Pickle all the documents in the index for faster future loading"""
    # Check if the index exists
    if not app.es_client.indices.exists(index=INDEX_NAME):
        print(f"Index {INDEX_NAME} does not exist. Cannot pickle.")
        return False
    
    # Get all documents from the index
    query = {"query": {"match_all": {}}, "size": 10000}  # Adjust size as needed
    
    try:
        # Initialize scroll
        result = app.es_client.search(
            index=INDEX_NAME,
            body=query,
            scroll="2m"  # Keep the search context alive for 2 minutes
        )
        
        # Get the scroll ID
        scroll_id = result["_scroll_id"]
        scroll_size = len(result["hits"]["hits"])
        
        # Store all documents
        all_docs = []
        
        # Get all documents
        while scroll_size > 0:
            all_docs.extend(result["hits"]["hits"])
            
            # Scroll to next batch
            result = app.es_client.scroll(scroll_id=scroll_id, scroll="2m")
            
            # Update scroll ID and size
            scroll_id = result["_scroll_id"]
            scroll_size = len(result["hits"]["hits"])
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(PICKLE_PATH), exist_ok=True)
        
        # Pickle the documents and index settings
        index_data = {
            "documents": all_docs,
            "settings": app.es_client.indices.get_settings(index=INDEX_NAME),
            "mappings": app.es_client.indices.get_mapping(index=INDEX_NAME),
            "timestamp": time.time()
        }
        
        with open(PICKLE_PATH, 'wb') as f:
            pickle.dump(index_data, f)
        
        print(f"Pickled {len(all_docs)} documents to {PICKLE_PATH}")
        return True
        
    except Exception as e:
        print(f"Error pickling index: {str(e)}")
        return False

def load_index_from_pickle():
    """Load the index from a pickle file"""
    if not os.path.exists(PICKLE_PATH):
        print(f"Pickle file not found at {PICKLE_PATH}")
        return False
    
    try:
        # Load the pickle file
        with open(PICKLE_PATH, 'rb') as f:
            index_data = pickle.load(f)
        
        print(f"Loaded pickle file from {PICKLE_PATH}")
        
        # Check if index exists and delete if it does
        if app.es_client.indices.exists(index=INDEX_NAME):
            app.es_client.indices.delete(index=INDEX_NAME)
            print(f"Deleted existing index {INDEX_NAME}")
        
        # Create the index with the same settings and mappings
        app.es_client.indices.create(
            index=INDEX_NAME,
            body={
                "settings": index_data["settings"][INDEX_NAME]["settings"],
                "mappings": index_data["mappings"][INDEX_NAME]["mappings"]
            }
        )
        print(f"Created index {INDEX_NAME} with pickled settings")
        
        # Process in chunks to avoid request size limits
        chunk_size = 1000  # Process 1000 records at a time
        docs = index_data["documents"]
        total_chunks = (len(docs) + chunk_size - 1) // chunk_size  # Ceiling division
        
        successful_chunks = 0
        
        for i in range(0, len(docs), chunk_size):
            chunk = docs[i:i+chunk_size]
            chunk_num = i // chunk_size + 1
            
            # Prepare bulk indexing operation for this chunk
            bulk_data = []
            for doc in chunk:
                bulk_data.append({"index": {"_index": INDEX_NAME, "_id": doc["_id"]}})
                bulk_data.append(doc["_source"])
            
            # Perform bulk indexing for this chunk
            try:
                if bulk_data:
                    app.es_client.bulk(body=bulk_data, refresh=(chunk_num == total_chunks))
                    successful_chunks += 1
                    print(f"Indexed chunk {chunk_num}/{total_chunks} ({len(chunk)} documents) from pickle")
            except Exception as e:
                print(f"Error indexing chunk {chunk_num} from pickle: {str(e)}")
        
        print(f"Completed indexing from pickle: {successful_chunks}/{total_chunks} chunks successful")
        return successful_chunks > 0
    
    except Exception as e:
        print(f"Error loading index from pickle: {str(e)}")
        return False

@app.route('/search', methods=['GET'])
def search():
    """API endpoint for searching recipes"""
    query = request.args.get('q', '')
    size = int(request.args.get('size', 10))
    
    if not query:
        return jsonify({"error": "Query parameter 'q' is required"}), 400
    
    # Perform BM25 search
    search_body = {
        "size": size,
        "query": {
            "multi_match": {
                "query": query,
                "fields": ["Name^3", "Description^2", "Keywords^2", "RecipeIngredientParts", "RecipeInstructions", "text"],
                "type": "best_fields"
            }
        },
        "highlight": {
            "fields": {
                "Name": {},
                "Description": {},
                "RecipeIngredientParts": {},
                "RecipeInstructions": {}
            }
        }
    }
    
    try:
        response = app.es_client.search(index=INDEX_NAME, body=search_body)
        hits = response["hits"]["hits"]
        
        results = []
        for hit in hits:
            source = hit["_source"]
            result = {
                "id": source.get("RecipeId", ""),
                "name": source.get("Name", ""),
                "description": source.get("Description", ""),
                "category": source.get("RecipeCategory", ""),
                "ingredients": source.get("RecipeIngredientParts", ""),
                "instructions": source.get("RecipeInstructions", ""),
                "rating": source.get("AggregatedRating", 0),
                "reviews": source.get("ReviewCount", 0),
                "image": source.get("Images", ""),
                "score": hit["_score"],
                "highlights": hit.get("highlight", {})
            }
            results.append(result)
        
        return jsonify({
            "total": response["hits"]["total"]["value"],
            "results": results
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/recipe/<recipe_id>')
def get_recipe(recipe_id):
    """API endpoint to get a specific recipe by ID"""
    try:
        search_body = {
            "query": {
                "term": {
                    "RecipeId": recipe_id
                }
            }
        }
        
        response = app.es_client.search(index=INDEX_NAME, body=search_body)
        hits = response["hits"]["hits"]
        
        if not hits:
            return jsonify({"error": "Recipe not found"}), 404
        
        source = hits[0]["_source"]
        recipe = {
            "id": source.get("RecipeId", ""),
            "name": source.get("Name", ""),
            "description": source.get("Description", ""),
            "category": source.get("RecipeCategory", ""),
            "ingredients": source.get("RecipeIngredientParts", ""),
            "instructions": source.get("RecipeInstructions", ""),
            "rating": source.get("AggregatedRating", 0),
            "reviews": source.get("ReviewCount", 0),
            "image": source.get("Images", "")
        }
        
        return jsonify(recipe)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/initialize', methods=['GET', 'POST'])
def initialize():
    """Initialize the Elasticsearch index and load data"""
    try:
        # Delete existing index if it exists to start clean
        if app.es_client.indices.exists(index=INDEX_NAME):
            app.es_client.indices.delete(index=INDEX_NAME)
            print(f"Deleted existing index {INDEX_NAME}")
            
        # Check if pickle file exists
        if os.path.exists(PICKLE_PATH):
            print("Pickle file found. Loading index from pickle...")
            success = load_index_from_pickle()
            if success:
                return jsonify({
                    "status": "success", 
                    "message": "Index loaded from pickle successfully"
                })
            else:
                # If loading from pickle fails, try creating from parquet
                print("Failed to load from pickle. Trying to create from parquet...")
        
        # Create from parquet
        create_index()
        success = load_data_from_parquet()
        
        if success:
            # After successful loading, pickle the index for future use
            pickle_success = pickle_index()
            pickle_msg = " and pickled for future use" if pickle_success else " but failed to pickle"
            
            return jsonify({
                "status": "success", 
                "message": f"Index created and data loaded successfully{pickle_msg}"
            })
        else:
            return jsonify({
                "status": "error", 
                "message": "Failed to load data from parquet"
            }), 500
    
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    # Check if index exists and initialize if needed
    if not app.es_client.indices.exists(index=INDEX_NAME):
        print("Index doesn't exist. Initializing...")
        initialize()
    else:
        print(f"Index {INDEX_NAME} already exists with {app.es_client.count(index=INDEX_NAME)['count']} documents")
    
    # Start the Flask server
    app.run(debug=False)