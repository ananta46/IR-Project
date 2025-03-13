import pandas as pd

# Path to your parquet file
parquet_path = 'backend/resource/csv/completed_recipes.parquet'

# Load the parquet file
df = pd.read_parquet(parquet_path)

# Check if the specific IDs exist
ids_to_check = ["38551", "33851", "29782", "30578", "47276", 
                "12345", "56789", "44732", "39725", "20619"]

print(f"Total recipes in parquet: {len(df)}")
print("Checking for specific IDs:")

for recipe_id in ids_to_check:
    # Assuming 'RecipeId' is the column name for recipe IDs
    if recipe_id in df['RecipeId'].values:
        print(f"Recipe ID {recipe_id} EXISTS in parquet")
    else:
        print(f"Recipe ID {recipe_id} NOT FOUND in parquet")

# Show the first few recipe IDs from the parquet file
print("\nSample Recipe IDs from parquet:")
print(df['RecipeId'].head(10).tolist())