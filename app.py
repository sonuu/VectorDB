# 1. Import required libraries
from transformers import AutoTokenizer, AutoModel
import torch

# 2. Set up your HuggingFace API key
HF_API_KEY = "Your_Key"

# 3. Function to convert text to vector embedding
def text_to_embedding(text, model_name="sentence-transformers/all-MiniLM-L6-v2"):
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    
    # Tokenize the input text
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    #Becomes Tokens (simplified): ["[CLS]", "I", "love", "to", "eat", "pizza", "[SEP]"] (7 tokens).
    
    # Get embeddings
    with torch.no_grad():
        outputs = model(**inputs)
        # Take the mean of the last hidden state to get sentence embedding
        embeddings = outputs.last_hidden_state.mean(dim=1)
    
    return embeddings

# 4. Main execution
if __name__ == "__main__":
    # Example text
    sample_text = "CAT"
    
    # Get embedding
    embedding = text_to_embedding(sample_text)
    
    # Print results
    print(f"Text: {sample_text}")
    print(f"Embedding shape: {embedding.shape}")
    print(f"Embedding: {embedding[0]}...")  # Show first 5 values
    #print(f"Embedding: {embedding[0][:5]}...")  # Show first 5 values