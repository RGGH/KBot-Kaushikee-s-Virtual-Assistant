from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login

# Authenticate with Hugging Face
login(token="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")  # Replace with your token

# Specify the model name
model_name = "meta-llama/Meta-Llama-3-8B-Instruct"

# Load and save the model and tokenizer
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Save the model and tokenizer to the local directory
model_directory = "local_model_directory"
model.save_pretrained(model_directory)
tokenizer.save_pretrained(model_directory)

print("Model and tokenizer have been downloaded and saved locally.")

