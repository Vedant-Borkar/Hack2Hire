from sentence_transformers import SentenceTransformer, util

# Load the pre-trained model
model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

# Example: Your answer vs user's answer
correct_answer = "The company should focus on customer retention strategies."
user_answer = "The business should work on keeping current customers angry."

# Convert sentences to embeddings
correct_embedding = model.encode(correct_answer, convert_to_tensor=True)
user_embedding = model.encode(user_answer, convert_to_tensor=True)

# Compute similarity score
similarity = util.pytorch_cos_sim(correct_embedding, user_embedding)
print("Similarity Score:", similarity.item())