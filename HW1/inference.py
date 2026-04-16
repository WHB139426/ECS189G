import torch
import os
from dataset import load_tiny_shakespeare
from model import TransformerLanguageModel

# ==========================================
# 1. Architecture Parameters (Must match train.py)
# ==========================================
context_length = 64
n_embd = 128          
n_head = 3           
n_layer = 3          
dropout = 0

device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f"🚀 Running inference on device: {device}")

# ==========================================
# 2. Load Vocabulary and Encoders
# ==========================================
_, _, vocab_size, encode, decode = load_tiny_shakespeare()

# ==========================================
# 3. Initialize Model 
# ==========================================
model = TransformerLanguageModel(
    vocab_size=vocab_size,
    context_length=context_length,
    n_embd=n_embd,
    n_head=n_head,
    n_layer=n_layer,
    dropout=dropout,
    device=device
)
model = model.to(device)
model.eval()

# ==========================================
# 4. Define Prompts
# ==========================================
max_new_tokens = 200
prompts = [
    "ROMEO:\n",
    "JULIET:\nO Romeo, Romeo!",
    "KING RICHARD III:\nWhat do you want?",
    "To be, or not to be, that is the question:"
]

def generate_responses(model_instance, tag=""):
    print("\n" + "="*50)
    print(f"🎭 {tag} 🎭")
    print("="*50)

    for idx, prompt_text in enumerate(prompts):
        print(f"\n--- Prompt {idx + 1} ---")
        print(f"Input: \n{prompt_text}")
        print("-" * 20)
        
        context_idx = torch.tensor(encode(prompt_text), dtype=torch.long, device=device).unsqueeze(0)
        
        with torch.no_grad():
            generated_idx = model_instance.generate(context_idx, max_new_tokens=max_new_tokens)
        
        generated_text = decode(generated_idx[0].tolist())
        
        print(f"Output (Input included): \n{generated_text}")
        print("="*50)

# ==========================================
# 5. Run Generation (Untrained vs Trained)
# ==========================================

# Phase 1: Output from the randomly initialized, UNTRAINED model
generate_responses(model, tag="UNTRAINED MODEL GENERATION (BASELINE)")

# Load the trained weights
checkpoint_path = os.path.join('checkpoints', 'best_model.pt')
if not os.path.exists(checkpoint_path):
    raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}. Please implement and run train.py first.")

model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
print(f"\n✅ Best model weights loaded from {checkpoint_path}")

# Phase 2: Output from the TRAINED model
generate_responses(model, tag="TRAINED MODEL GENERATION")