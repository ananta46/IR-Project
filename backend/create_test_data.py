import sqlite3
from werkzeug.security import generate_password_hash
import os

# Database configuration
DB_FILE = "food_app.db"

def create_test_data():
    """Create test users, folders, and bookmarks"""
    # Check if database exists
    if not os.path.exists(DB_FILE):
        print(f"Database file {DB_FILE} not found. Please run your app first to initialize the database.")
        return
    
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    try:
        # Create test users
        test_users = [
            {
                "username": "test_user1",
                "email": "test1@example.com",
                "password": "password123"
            },
            {
                "username": "test_user2",
                "email": "test2@example.com",
                "password": "password123"
            }
        ]
        
        user_ids = []
        
        for user in test_users:
            # Check if user already exists
            cursor.execute("SELECT id FROM users WHERE username = ?", (user["username"],))
            existing_user = cursor.fetchone()
            
            if existing_user:
                print(f"User {user['username']} already exists.")
                user_ids.append(existing_user["id"])
            else:
                # Create new user
                cursor.execute(
                    "INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)",
                    (user["username"], user["email"], generate_password_hash(user["password"]))
                )
                user_id = cursor.lastrowid
                user_ids.append(user_id)
                print(f"Created user {user['username']} with ID {user_id}")
        
        conn.commit()
        
        # Create test folders for each user
        for i, user_id in enumerate(user_ids):
            # Check if user already has folders
            cursor.execute("SELECT COUNT(*) as count FROM folders WHERE user_id = ?", (user_id,))
            folder_count = cursor.fetchone()["count"]
            
            if folder_count > 0:
                print(f"User {test_users[i]['username']} already has folders. Skipping folder creation.")
                continue
            
            # Create folders
            folders = [
                "Favorites",
                "Breakfast Ideas",
                "Quick Dinners",
                "Desserts"
            ]
            
            folder_ids = []
            
            for folder_name in folders:
                cursor.execute(
                    "INSERT INTO folders (user_id, name) VALUES (?, ?)",
                    (user_id, folder_name)
                )
                folder_id = cursor.lastrowid
                folder_ids.append(folder_id)
                print(f"Created folder '{folder_name}' with ID {folder_id} for user {test_users[i]['username']}")
            
            conn.commit()
            
            # Add some test bookmarks with recipe IDs
            # Note: These IDs should exist in your Elasticsearch index
            test_recipe_ids = [
                "38551", "33851", "29782", "30578", "47276", 
                "12345", "56789", "44732", "39725", "20619"
            ]
            
            for j, folder_id in enumerate(folder_ids):
                # Add 2-3 recipes per folder
                for k in range(min(3, len(test_recipe_ids))):
                    recipe_idx = (j * 3 + k) % len(test_recipe_ids)
                    recipe_id = test_recipe_ids[recipe_idx]
                    rating = (k % 5) + 1  # Rating from 1-5
                    
                    # Check if bookmark already exists
                    cursor.execute(
                        "SELECT id FROM bookmarks WHERE user_id = ? AND folder_id = ? AND recipe_id = ?",
                        (user_id, folder_id, recipe_id)
                    )
                    existing_bookmark = cursor.fetchone()
                    
                    if existing_bookmark:
                        print(f"Bookmark for recipe {recipe_id} already exists in folder {folder_id}")
                    else:
                        cursor.execute(
                            "INSERT INTO bookmarks (user_id, folder_id, recipe_id, rating) VALUES (?, ?, ?, ?)",
                            (user_id, folder_id, recipe_id, rating)
                        )
                        bookmark_id = cursor.lastrowid
                        print(f"Created bookmark for recipe {recipe_id} with rating {rating} in folder {folder_id}")
            
            conn.commit()
        
        print("Test data creation complete!")
        
    except Exception as e:
        conn.rollback()
        print(f"Error creating test data: {str(e)}")
    
    finally:
        conn.close()

if __name__ == "__main__":
    create_test_data()