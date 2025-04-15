### Example Pipeline: Use BERT to Extract Column Matches, Then Mistral to Plan Task

# 1. Load SentenceTransformer model (from Hugging Face)
from sentence_transformers import SentenceTransformer, util
model = SentenceTransformer("all-MiniLM-L6-v2")

# 2. User prompt and dataset columns
user_prompt = "Show distribution of income by city"
columns = ["City", "Income", "Gender", "Education", "Zip Code"]

# 3. Find the most relevant columns using cosine similarity
query_embedding = model.encode(user_prompt, convert_to_tensor=True)
column_embeddings = model.encode(columns, convert_to_tensor=True)
cosine_scores = util.pytorch_cos_sim(query_embedding, column_embeddings)[0]

# 4. Get top matches
top_indices = cosine_scores.argsort(descending=True)[:2]
matched_cols = [columns[i] for i in top_indices]

# 5. Build rewritten query using Mistral
from utils import mistral_generate
mistral_prompt = f"Given this context: columns = {matched_cols}, user wants: '{user_prompt}', generate a clear plotting instruction."
revised_query = mistral_generate(mistral_prompt)

# OUTPUT: revised_query → passed to chart agent
print(revised_query)