from flask import Flask, request, jsonify, session
from elasticsearch import Elasticsearch
import pandas as pd
import pickle
import os
import time
import sqlite3
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps

import numpy as np
import faiss
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json

from recommendation import RecipeRecommender

# Custom JSON encoder to handle numpy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)



app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Change this in production!
app.json_encoder = NumpyEncoder

# Database configuration
DB_FILE = "food_app.db"

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


# Initialize the recommender at app startup
app.recommender = None



def init_recommender():
    """Initialize the recommender system"""
    try:
        print("Recommender system initializing")
        app.recommender = RecipeRecommender()
        # Sync with Elasticsearch to check ID compatibility
        app.recommender.sync_with_elasticsearch(app.es_client, INDEX_NAME)
        print("Recommender system initialized successfully")
    except Exception as e:
        print(f"Error initializing recommender system: {str(e)}")


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

# SQLite Helper Functions
def get_db_connection():
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Initialize the SQLite database with required tables"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Create users table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        email TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    # Create folders table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS folders (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        name TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users (id)
    )
    ''')
    
    # Create bookmarks table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS bookmarks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        folder_id INTEGER NOT NULL,
        recipe_id TEXT NOT NULL,
        rating INTEGER CHECK(rating >= 1 AND rating <= 5),
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users (id),
        FOREIGN KEY (folder_id) REFERENCES folders (id)
    )
    ''')
    
    conn.commit()
    conn.close()
    print("SQLite database initialized successfully!")

# Authentication Decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return jsonify({"error": "Authentication required"}), 401
        return f(*args, **kwargs)
    return decorated_function

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

# Authentication Routes
@app.route('/register', methods=['POST'])
def register():
    data = request.get_json()
    username = data.get('username')
    email = data.get('email')
    password = data.get('password')
    
    if not username or not email or not password:
        return jsonify({"error": "Missing required fields"}), 400
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Check if user already exists
    cursor.execute("SELECT * FROM users WHERE username = ? OR email = ?", (username, email))
    existing_user = cursor.fetchone()
    
    if existing_user:
        conn.close()
        return jsonify({"error": "Username or email already exists"}), 409
    
    # Hash the password
    password_hash = generate_password_hash(password)
    
    try:
        # Insert new user
        cursor.execute(
            "INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)",
            (username, email, password_hash)
        )
        user_id = cursor.lastrowid
        conn.commit()
        
        # Create a default folder for the user
        cursor.execute(
            "INSERT INTO folders (user_id, name) VALUES (?, ?)",
            (user_id, "Favorites")
        )
        conn.commit()
        
        conn.close()
        return jsonify({"id": user_id, "message": "User registered successfully"}), 201
    
    except Exception as e:
        conn.close()
        return jsonify({"error": str(e)}), 500

@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')
    
    if not username or not password:
        return jsonify({"error": "Missing username or password"}), 400
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Get user
    cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
    user = cursor.fetchone()
    conn.close()
    
    if not user or not check_password_hash(user['password_hash'], password):
        return jsonify({"error": "Invalid username or password"}), 401
    
    # Set session
    session['user_id'] = user['id']
    session['username'] = user['username']
    
    return jsonify({
        "message": "Login successful", 
        "user": {
            "id": user['id'], 
            "username": user['username']
        }
    }), 200

@app.route('/logout', methods=['POST'])
def logout():
    session.pop('user_id', None)
    session.pop('username', None)
    return jsonify({"message": "Logged out successfully"}), 200

@app.route('/profile', methods=['GET'])
@login_required
def profile():
    user_id = session['user_id']
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute(
        "SELECT id, username, email, created_at FROM users WHERE id = ?", 
        (user_id,)
    )
    user = cursor.fetchone()
    
    if not user:
        conn.close()
        return jsonify({"error": "User not found"}), 404
    
    user_data = dict(user)
    
    # Get user's folders
    cursor.execute("SELECT * FROM folders WHERE user_id = ?", (user_id,))
    folders = [dict(row) for row in cursor.fetchall()]
    
    # Get bookmark counts for each folder
    for folder in folders:
        cursor.execute(
            "SELECT COUNT(*) as count FROM bookmarks WHERE folder_id = ?", 
            (folder['id'],)
        )
        count = cursor.fetchone()
        folder['bookmark_count'] = count['count'] if count else 0
    
    conn.close()
    
    return jsonify({
        "user": user_data,
        "folders": folders
    }), 200

# Folder Management Routes
@app.route('/folders', methods=['GET'])
@login_required
def get_folders():
    user_id = session['user_id']
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT * FROM folders WHERE user_id = ?", (user_id,))
    folders = [dict(row) for row in cursor.fetchall()]
    
    conn.close()
    
    return jsonify(folders), 200

@app.route('/folders', methods=['POST'])
@login_required
def create_folder():
    user_id = session['user_id']
    data = request.get_json()
    folder_name = data.get('name')
    
    if not folder_name:
        return jsonify({"error": "Folder name is required"}), 400
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute(
            "INSERT INTO folders (user_id, name) VALUES (?, ?)",
            (user_id, folder_name)
        )
        folder_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return jsonify({
            "id": folder_id, 
            "name": folder_name, 
            "message": "Folder created successfully"
        }), 201
    
    except Exception as e:
        conn.close()
        return jsonify({"error": str(e)}), 500

@app.route('/folders/<int:folder_id>', methods=['DELETE'])
@login_required
def delete_folder(folder_id):
    user_id = session['user_id']
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Check if folder exists and belongs to user
    cursor.execute(
        "SELECT * FROM folders WHERE id = ? AND user_id = ?", 
        (folder_id, user_id)
    )
    folder = cursor.fetchone()
    
    if not folder:
        conn.close()
        return jsonify({"error": "Folder not found or not owned by user"}), 404
    
    try:
        # Delete all bookmarks in this folder
        cursor.execute("DELETE FROM bookmarks WHERE folder_id = ?", (folder_id,))
        
        # Delete the folder
        cursor.execute("DELETE FROM folders WHERE id = ?", (folder_id,))
        
        conn.commit()
        conn.close()
        
        return jsonify({"message": "Folder deleted successfully"}), 200
    
    except Exception as e:
        conn.close()
        return jsonify({"error": str(e)}), 500

# Bookmark Management Routes
@app.route('/bookmarks', methods=['POST'])
@login_required
def bookmark_recipe():
    user_id = session['user_id']
    data = request.get_json()
    
    folder_id = data.get('folder_id')
    recipe_id = data.get('recipe_id')
    rating = data.get('rating')
    
    if not folder_id or not recipe_id:
        return jsonify({"error": "Folder ID and recipe ID are required"}), 400
    
    # Validate rating
    if rating and (rating < 1 or rating > 5):
        return jsonify({"error": "Rating must be between 1 and 5"}), 400
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Check if folder exists and belongs to user
    cursor.execute(
        "SELECT * FROM folders WHERE id = ? AND user_id = ?", 
        (folder_id, user_id)
    )
    folder = cursor.fetchone()
    
    if not folder:
        conn.close()
        return jsonify({"error": "Folder not found or not owned by user"}), 404
    
    try:
        # Check if bookmark already exists
        cursor.execute(
            "SELECT * FROM bookmarks WHERE user_id = ? AND recipe_id = ? AND folder_id = ?",
            (user_id, recipe_id, folder_id)
        )
        existing = cursor.fetchone()
        
        if existing:
            # Update rating if bookmark exists
            cursor.execute(
                "UPDATE bookmarks SET rating = ? WHERE id = ?",
                (rating, existing['id'])
            )
            message = "Recipe rating updated successfully"
        else:
            # Create new bookmark
            cursor.execute(
                "INSERT INTO bookmarks (user_id, folder_id, recipe_id, rating) VALUES (?, ?, ?, ?)",
                (user_id, folder_id, recipe_id, rating)
            )
            message = "Recipe bookmarked successfully"
        
        conn.commit()
        conn.close()
        
        return jsonify({"message": message}), 200
    
    except Exception as e:
        conn.close()
        return jsonify({"error": str(e)}), 500

@app.route('/folders/<int:folder_id>/bookmarks', methods=['GET'])
@login_required
def get_bookmarks(folder_id):
    user_id = session['user_id']
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Check if folder exists and belongs to user
    cursor.execute(
        "SELECT * FROM folders WHERE id = ? AND user_id = ?", 
        (folder_id, user_id)
    )
    folder = cursor.fetchone()
    
    if not folder:
        conn.close()
        return jsonify({"error": "Folder not found or not owned by user"}), 404
    
    # Get bookmarks
    cursor.execute(
        "SELECT * FROM bookmarks WHERE user_id = ? AND folder_id = ? ORDER BY created_at DESC",
        (user_id, folder_id)
    )
    bookmarks = [dict(row) for row in cursor.fetchall()]
    
    # Get recipe details for each bookmark
    recipe_ids = [bookmark['recipe_id'] for bookmark in bookmarks]
    recipes = []
    
    for recipe_id in recipe_ids:
        # Get recipe from Elasticsearch
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
            
            if hits:
                source = hits[0]["_source"]
                recipes.append({
                    "id": source.get("RecipeId", ""),
                    "name": source.get("Name", ""),
                    "description": source.get("Description", ""),
                    "category": source.get("RecipeCategory", ""),
                    "image": source.get("Images", ""),
                    "rating": source.get("AggregatedRating", 0),
                })
        except:
            # If we can't get the recipe, continue to the next one
            continue
    
    conn.close()
    
    return jsonify({
        "folder": dict(folder),
        "bookmarks": bookmarks,
        "recipes": recipes
    }), 200

@app.route('/bookmarks/<int:bookmark_id>', methods=['DELETE'])
@login_required
def delete_bookmark(bookmark_id):
    user_id = session['user_id']
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Check if bookmark exists and belongs to user
    cursor.execute(
        "SELECT * FROM bookmarks WHERE id = ? AND user_id = ?", 
        (bookmark_id, user_id)
    )
    bookmark = cursor.fetchone()
    
    if not bookmark:
        conn.close()
        return jsonify({"error": "Bookmark not found or not owned by user"}), 404
    
    try:
        # Delete the bookmark
        cursor.execute("DELETE FROM bookmarks WHERE id = ?", (bookmark_id,))
        conn.commit()
        conn.close()
        
        return jsonify({"message": "Bookmark deleted successfully"}), 200
    
    except Exception as e:
        conn.close()
        return jsonify({"error": str(e)}), 500


@app.route('/recipe/<recipe_id>/suggestions', methods=['GET'])
def get_recipe_suggestions(recipe_id):
    """Get recipe suggestions similar to a specific recipe"""
    try:
        # Initialize recommender if needed
        if app.recommender is None:
            init_recommender()
            
        if app.recommender is None:
            return jsonify({
                "error": "Recommender system not available"
            }), 500
        
        # Get similar recipes
        similar_recipes = app.recommender.get_recommendations_by_recipe_id(
            recipe_id, 
            num_recommendations=8
        )
        
        return jsonify({
            "suggestions": similar_recipes
        }), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/recommendations', methods=['GET'])
@login_required
def get_recommendations():
    """Get personalized recommendations for the current user"""
    user_id = session['user_id']
    
    # Get optional folder parameter
    folder_id = request.args.get('folder_id', None)
    
    try:
        # Initialize the recommender if not already initialized
        if app.recommender is None:
            init_recommender()
            
        # If the recommender still couldn't be initialized, return an error
        if app.recommender is None:
            return jsonify({
                "error": "Recommender system not available"
            }), 500
            
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Get user's bookmarks
        if folder_id:
            # Get bookmarks from a specific folder
            cursor.execute(
                "SELECT recipe_id, rating FROM bookmarks WHERE user_id = ? AND folder_id = ?", 
                (user_id, folder_id)
            )
        else:
            # Get all user's bookmarks
            cursor.execute(
                "SELECT recipe_id, rating FROM bookmarks WHERE user_id = ?", 
                (user_id,)
            )
            
        bookmarks = cursor.fetchall()
        # Convert recipe_ids to strings for consistent type matching
        bookmarked_recipe_ids = [str(bookmark['recipe_id']) for bookmark in bookmarks]
        
        # Close database connection
        conn.close()
        
        # Print debug info
        print(f"Bookmark IDs for recommender: {bookmarked_recipe_ids}")
        
        # If user has no bookmarks, return popular recipes
        if not bookmarks:
            popular_recipes = app.recommender.get_popular_recommendations(num_recommendations=8)
            random_recipes = app.recommender.get_random_recommendations(num_recommendations=4)
            
            return jsonify({
                "summary": {
                    "title": "Recipes you might like",
                    "recipes": popular_recipes
                },
                "random": {
                    "title": "Discover something new",
                    "recipes": random_recipes
                }
            }), 200
        
        # Get similarity-based recommendations using FAISS
        similar_recipes = app.recommender.get_recommendations_by_bookmarks(
            bookmarked_recipe_ids, 
            num_recommendations=10
        )
        
        # Get random recommendations
        random_recipes = app.recommender.get_random_recommendations(
            exclude_recipe_ids=bookmarked_recipe_ids,
            num_recommendations=5
        )
        
        # Prepare the response based on the folder_id parameter
        if folder_id:
            # Get the folder name
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM folders WHERE id = ?", (folder_id,))
            folder = cursor.fetchone()
            conn.close()
            
            folder_name = folder['name'] if folder else "This folder"
            
            return jsonify({
                "folder_recommendations": {
                    "title": f"More recipes like those in {folder_name}",
                    "recipes": similar_recipes
                },
                "discovery": {
                    "title": "Discover something new",
                    "recipes": random_recipes
                }
            }), 200
        else:
            # All bookmarks recommendations
            return jsonify({
                "summary": {
                    "title": "Based on your bookmarks",
                    "recipes": similar_recipes[:8]
                },
                "random": {
                    "title": "Discover something new",
                    "recipes": random_recipes
                }
            }), 200
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@app.route('/debug/recipes', methods=['GET'])
def debug_recipes():
    """Debug endpoint to check recipe IDs"""
    recipe_ids = request.args.get('ids', '').split(',')
    
    if not recipe_ids or recipe_ids[0] == '':
        # Return some random recipe IDs
        try:
            search_body = {
                "size": 10,
                "_source": ["RecipeId", "Name"],
                "query": {"match_all": {}}
            }
            
            response = app.es_client.search(index=INDEX_NAME, body=search_body)
            hits = response["hits"]["hits"]
            
            results = []
            for hit in hits:
                source = hit["_source"]
                results.append({
                    "id": source.get("RecipeId", ""),
                    "name": source.get("Name", "")
                })
            
            return jsonify({
                "message": "Random recipe IDs from Elasticsearch",
                "recipes": results
            })
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    
    # Check specific recipe IDs
    results = []
    for recipe_id in recipe_ids:
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
            
            if hits:
                source = hits[0]["_source"]
                results.append({
                    "id": recipe_id,
                    "found": True,
                    "name": source.get("Name", "")
                })
            else:
                results.append({
                    "id": recipe_id,
                    "found": False
                })
        except Exception as e:
            results.append({
                "id": recipe_id,
                "found": False,
                "error": str(e)
            })
    
    return jsonify({
        "results": results
    })


if __name__ == '__main__':
    # Check if database file exists
    if not os.path.exists(DB_FILE):
        print("Database doesn't exist. Initializing...")
        init_db()
    else:
        print(f"Database {DB_FILE} already exists")
    
    # Check if Elasticsearch index exists and initialize if needed
    if not app.es_client.indices.exists(index=INDEX_NAME):
        print("Index doesn't exist. Initializing...")
        initialize()
    else:
        print(f"Index {INDEX_NAME} already exists with {app.es_client.count(index=INDEX_NAME)['count']} documents")
    
    # Initialize the recommender system in a separate thread to avoid blocking startup
    import threading
    threading.Thread(target=init_recommender).start()
    
    # Start the Flask server
    app.run(debug=False)