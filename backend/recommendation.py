import numpy as np
import pandas as pd
import faiss
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

# Download NLTK resources (run once)
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

def get_all_elasticsearch_recipe_ids(es_client, index_name):
    """Get all recipe IDs from Elasticsearch using scroll API"""
    print("Retrieving all recipe IDs from Elasticsearch...")
    
    try:
        # Initialize the scroll
        scroll_body = {
            "_source": ["RecipeId"],
            "query": {"match_all": {}},
            "size": 1000  # Process 1000 documents at a time
        }
        
        # Get the first batch and scroll ID
        response = es_client.search(
            index=index_name,
            body=scroll_body,
            scroll="2m"  # Keep the search context alive for 2 minutes
        )
        
        # Get the scroll ID
        scroll_id = response["_scroll_id"]
        scroll_size = len(response["hits"]["hits"])
        
        # Collect all recipe IDs
        recipe_ids = []
        
        # Process initial batch
        for hit in response["hits"]["hits"]:
            if "RecipeId" in hit["_source"]:
                recipe_ids.append(hit["_source"]["RecipeId"])
        
        # Continue scrolling until no more hits
        while scroll_size > 0:
            response = es_client.scroll(scroll_id=scroll_id, scroll="2m")
            
            # Process this batch of results
            for hit in response["hits"]["hits"]:
                if "RecipeId" in hit["_source"]:
                    recipe_ids.append(hit["_source"]["RecipeId"])
            
            # Update scroll ID and size
            scroll_id = response["_scroll_id"]
            scroll_size = len(response["hits"]["hits"])
        
        print(f"Retrieved {len(recipe_ids)} recipe IDs from Elasticsearch")
        return recipe_ids
        
    except Exception as e:
        print(f"Error retrieving recipe IDs from Elasticsearch: {str(e)}")
        return []

class RecipeRecommender:
    def __init__(self, parquet_path=None):
        # Get the base directory where the script is located
        base_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Default paths with absolute references
        if parquet_path is None:
            self.parquet_path = os.path.join(base_dir, 'resource', 'csv', 'completed_recipes.parquet')
        else:
            self.parquet_path = parquet_path
            
        self.index_path = os.path.join(base_dir, 'resource', 'pickles', 'faiss_index.pkl')
        self.vectorizer_path = os.path.join(base_dir, 'resource', 'pickles', 'tfidf_vectorizer.pkl')
        self.recipe_ids_path = os.path.join(base_dir, 'resource', 'pickles', 'recipe_ids.pkl')
        self.df_path = os.path.join(base_dir, 'resource', 'pickles', 'recipe_df.pkl')
        
        # Initialize FAISS index and other components
        self.faiss_index = None
        self.vectorizer = None
        self.recipe_ids = None
        self.df = None
        
        # Load or build resources
        self.initialize()
    
    def initialize(self):
        """Initialize the recommender by loading or building the FAISS index"""
        # Check if pre-computed resources exist
        if (os.path.exists(self.index_path) and 
            os.path.exists(self.vectorizer_path) and 
            os.path.exists(self.recipe_ids_path) and
            os.path.exists(self.df_path)):
            
            print("Loading pre-computed recommendation resources...")
            self.load_resources()
            
            # Verify the recipe IDs exist in both FAISS and Elasticsearch
            print(f"FAISS index has {len(self.recipe_ids)} recipes")
            
            # Randomly check a few recipe IDs to see if they're found
            if len(self.recipe_ids) > 0:
                sample_ids = self.recipe_ids[:5]
                print(f"Sample recipe IDs from FAISS: {sample_ids}")
        else:
            print("Building recommendation resources from scratch...")
            self.build_resources()
    
    def load_resources(self):
        """Load pre-computed FAISS index and related resources"""
        with open(self.index_path, 'rb') as f:
            self.faiss_index = pickle.load(f)
        
        with open(self.vectorizer_path, 'rb') as f:
            self.vectorizer = pickle.load(f)
        
        with open(self.recipe_ids_path, 'rb') as f:
            self.recipe_ids = pickle.load(f)
            
        with open(self.df_path, 'rb') as f:
            self.df = pickle.load(f)
    
    def build_resources(self):
        """Build FAISS index and related resources from the recipe data"""
        # Load the dataset
        self.df = pd.read_parquet(self.parquet_path)
        
        # Convert RecipeId to string if it's not already
        self.df['RecipeId'] = self.df['RecipeId'].astype(str)
        
        # Preprocess text for better recommendations
        self.df['combined_features'] = self.df.apply(self._combine_features, axis=1)
        
        # Create TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        # Create TF-IDF matrix
        tfidf_matrix = self.vectorizer.fit_transform(self.df['combined_features'].fillna(''))
        
        # Convert to dense numpy array for FAISS
        feature_vectors = tfidf_matrix.toarray().astype('float32')
        
        # Store recipe IDs in the same order as the feature vectors
        self.recipe_ids = self.df['RecipeId'].values
        
        # Explicitly ensure recipe_ids are strings
        self.recipe_ids = np.array([str(id) for id in self.recipe_ids])
        
        # Build FAISS index
        # Use L2 distance (convert from cosine similarity later)
        dimension = feature_vectors.shape[1]
        self.faiss_index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        
        # Normalize vectors for cosine similarity
        faiss.normalize_L2(feature_vectors)
        
        # Add vectors to the index
        self.faiss_index.add(feature_vectors)
        
        # Save resources for future use
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        
        with open(self.index_path, 'wb') as f:
            pickle.dump(self.faiss_index, f)
        
        with open(self.vectorizer_path, 'wb') as f:
            pickle.dump(self.vectorizer, f)
        
        with open(self.recipe_ids_path, 'wb') as f:
            pickle.dump(self.recipe_ids, f)
            
        with open(self.df_path, 'wb') as f:
            pickle.dump(self.df, f)
    
    def _combine_features(self, row):
        """Combine relevant features for similarity calculation"""
        features = []
        
        # Add recipe name (most important)
        if not pd.isna(row['Name']):
            features.append(row['Name'] * 3)  # Repeat for more weight
        
        # Add ingredients
        if not pd.isna(row['RecipeIngredientParts']):
            features.append(row['RecipeIngredientParts'])
        
        # Add description
        if not pd.isna(row['Description']):
            features.append(row['Description'])
        
        # Add keywords
        if not pd.isna(row['Keywords']):
            features.append(row['Keywords'])
        
        # Combine all features
        return ' '.join(features)
    
    def _preprocess_text(self, text):
        """Preprocess text for better matching"""
        if pd.isna(text) or text == '':
            return ''
        
        # Convert to lowercase
        text = text.lower()
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and stem
        stop_words = set(stopwords.words('english'))
        stemmer = PorterStemmer()
        
        tokens = [stemmer.stem(word) for word in tokens if word.isalnum() and word not in stop_words]
        
        return ' '.join(tokens)
    
    # In recommendation.py - update this method in your RecipeRecommender class
    def get_recommendations_by_recipe_id(self, recipe_id, num_recommendations=10):
        """Get recommendations based on a specific recipe ID"""
        try:
            # Convert recipe_id to string to ensure type matching
            recipe_id = str(recipe_id)
            
            # Find the index of the recipe in our dataset
            recipe_idx = np.where(np.array([str(id) for id in self.recipe_ids]) == recipe_id)[0]
            
            if len(recipe_idx) == 0:
                print(f"Recipe ID {recipe_id} not found in the FAISS index, although it exists in Elasticsearch")
                # Return random recommendations instead
                return self.get_random_recommendations(num_recommendations=num_recommendations)
            
            # Get the feature vector for this recipe
            query_vector = self.faiss_index.reconstruct(int(recipe_idx[0])).reshape(1, -1)
            
            # Search for similar recipes
            distances, indices = self.faiss_index.search(query_vector, num_recommendations + 1)
            
            # Filter out the query recipe itself
            recommendations = []
            for i, idx in enumerate(indices[0]):
                if str(self.recipe_ids[idx]) != recipe_id:
                    recipe_data = self.df.iloc[idx]
                    recommendations.append({
                        'id': recipe_data['RecipeId'],
                        'name': recipe_data['Name'],
                        'description': recipe_data['Description'] if not pd.isna(recipe_data['Description']) else '',
                        'image': recipe_data['Images'] if not pd.isna(recipe_data['Images']) else '',
                        'rating': recipe_data['AggregatedRating'] if not pd.isna(recipe_data['AggregatedRating']) else 0,
                        'similarity_score': float(distances[0][i]),
                        'type': 'similarity'
                    })
                    
                    if len(recommendations) >= num_recommendations:
                        break
            
            return recommendations
            
        except Exception as e:
            print(f"Error getting recommendations by recipe ID: {str(e)}")
            return self.get_random_recommendations(num_recommendations=num_recommendations)

    def get_recommendations_by_bookmarks(self, bookmarked_recipe_ids, num_recommendations=10):
        """Get recommendations based on a user's bookmarked recipes"""
        try:
            if not bookmarked_recipe_ids:
                return []
            
            # Find the indices of the bookmarked recipes that exist in our index
            found_recipe_ids = []
            for recipe_id in bookmarked_recipe_ids:
                # Ensure recipe_id is a string
                recipe_id = str(recipe_id)
                # Check if it's in the index
                if recipe_id in [str(id) for id in self.recipe_ids]:
                    found_recipe_ids.append(recipe_id)
                else:
                    print(f"Recipe ID {recipe_id} not found in the index")
            
            # If none of the bookmarked recipes are found, return random recommendations
            if not found_recipe_ids:
                print("None of the bookmarked recipes were found in the index. Returning random recommendations.")
                return self.get_random_recommendations(num_recommendations=num_recommendations)
            
            # Get recommendations for the recipes that were found
            all_recommendations = []
            for recipe_id in found_recipe_ids:
                recommendations = self.get_recommendations_by_recipe_id(recipe_id, num_recommendations=5)
                all_recommendations.extend(recommendations)
            
            # Remove duplicates and sort by similarity score
            unique_recipes = {}
            for rec in all_recommendations:
                if rec['id'] not in unique_recipes and rec['id'] not in bookmarked_recipe_ids:
                    unique_recipes[rec['id']] = rec
            
            # Sort by similarity score
            sorted_recommendations = sorted(
                unique_recipes.values(), 
                key=lambda x: x['similarity_score'] if 'similarity_score' in x else 0, 
                reverse=True
            )
            
            return sorted_recommendations[:num_recommendations]
            
        except Exception as e:
            print(f"Error getting recommendations by bookmarks: {str(e)}")
            return self.get_random_recommendations(num_recommendations=num_recommendations)
    
    def get_random_recommendations(self, exclude_recipe_ids=None, num_recommendations=5):
        """Get random recommendations"""
        if exclude_recipe_ids is None:
            exclude_recipe_ids = []
        
        try:
            # Filter out excluded recipes
            available_indices = [
                i for i, recipe_id in enumerate(self.recipe_ids) 
                if recipe_id not in exclude_recipe_ids
            ]
            
            if len(available_indices) <= num_recommendations:
                selected_indices = available_indices
            else:
                # Randomly select indices
                selected_indices = np.random.choice(
                    available_indices,
                    size=num_recommendations,
                    replace=False
                )
            
            # Get the selected recipes
            recommendations = []
            for idx in selected_indices:
                recipe_data = self.df.iloc[idx]
                recommendations.append({
                    'id': recipe_data['RecipeId'],
                    'name': recipe_data['Name'],
                    'description': recipe_data['Description'] if not pd.isna(recipe_data['Description']) else '',
                    'image': recipe_data['Images'] if not pd.isna(recipe_data['Images']) else '',
                    'rating': recipe_data['AggregatedRating'] if not pd.isna(recipe_data['AggregatedRating']) else 0,
                    'type': 'random'
                })
            
            return recommendations
        
        except Exception as e:
            print(f"Error getting random recommendations: {str(e)}")
            return []
    
    def get_popular_recommendations(self, exclude_recipe_ids=None, num_recommendations=5):
        """Get popular recommendations based on rating and review count"""
        if exclude_recipe_ids is None:
            exclude_recipe_ids = []
        
        try:
            # Create a copy of the dataframe to avoid modifying the original
            temp_df = self.df.copy()
            
            # Filter out excluded recipes
            temp_df = temp_df[~temp_df['RecipeId'].isin(exclude_recipe_ids)]
            
            # Sort by rating and review count
            temp_df['AggregatedRating'] = temp_df['AggregatedRating'].fillna(0)
            temp_df['ReviewCount'] = temp_df['ReviewCount'].fillna(0)
            
            # Calculate a popularity score
            temp_df['popularity'] = temp_df['AggregatedRating'] * np.log1p(temp_df['ReviewCount'])
            
            # Sort by popularity
            temp_df = temp_df.sort_values('popularity', ascending=False).head(num_recommendations)
            
            # Convert to list of dictionaries
            recommendations = []
            for _, row in temp_df.iterrows():
                recommendations.append({
                    'id': row['RecipeId'],
                    'name': row['Name'],
                    'description': row['Description'] if not pd.isna(row['Description']) else '',
                    'image': row['Images'] if not pd.isna(row['Images']) else '',
                    'rating': row['AggregatedRating'] if not pd.isna(row['AggregatedRating']) else 0,
                    'type': 'popular'
                })
            
            return recommendations
        
        except Exception as e:
            print(f"Error getting popular recommendations: {str(e)}")
            return []
        
    def sync_with_elasticsearch(self, es_client, index_name):
        """Sync recipe IDs between Elasticsearch and FAISS"""
        print("Syncing recipe IDs between Elasticsearch and FAISS...")
        
        try:
            # Get all recipe IDs from Elasticsearch using the scroll API
            es_recipe_ids = get_all_elasticsearch_recipe_ids(es_client, index_name)
            
            if not es_recipe_ids:
                print("Failed to retrieve recipe IDs from Elasticsearch")
                return False
                
            print(f"Found {len(es_recipe_ids)} recipe IDs in Elasticsearch")
            
            # Check overlap with FAISS
            faiss_recipe_ids = set(self.recipe_ids)
            es_recipe_ids_set = set(es_recipe_ids)
            
            common_ids = faiss_recipe_ids.intersection(es_recipe_ids_set)
            print(f"FAISS and Elasticsearch have {len(common_ids)} recipes in common")
            
            if len(common_ids) < 100:  # Arbitrary threshold
                print("WARNING: Very few recipes in common between FAISS and Elasticsearch")
                print("Consider rebuilding your FAISS index from the same data source")
            
            # Print some sample IDs from both sources
            print("\nSample FAISS IDs:", list(faiss_recipe_ids)[:5])
            print("Sample ES IDs:", list(es_recipe_ids_set)[:5])
            
            return True
        except Exception as e:
            print(f"Error syncing with Elasticsearch: {str(e)}")
            return False