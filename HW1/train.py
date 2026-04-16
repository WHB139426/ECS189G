import torch
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

from dataset import load_tiny_shakespeare, get_batch
from model import TransformerLanguageModel

# ==========================================
# 1. Hyperparameters
# ==========================================
batch_size = 64
max_iters = 5000     
eval_interval = 100  
learning_rate = 9e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 100        

# Model architecture parameters
context_length = 64
n_embd = 128          
n_head = 3           
n_layer = 3          
dropout = 0

print(f"🚀 Training on device: {device}")

# ==========================================
# 2. Data Preparation
# ==========================================
train_data, val_data, vocab_size, _, _ = load_tiny_shakespeare()
print(f"Vocab size (characters): {vocab_size}")

# ==========================================
# 3. Helper Functions
# ==========================================
def save_loss_plot(all_train_losses, val_iters, val_losses, save_path='loss_curve.png'):
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(all_train_losses)), all_train_losses, label='Train Loss (per batch)', color='tab:blue', alpha=0.3)
    plt.plot(val_iters, val_losses, label='Validation Loss', color='tab:orange', marker='o', linewidth=2)
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Curve')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(save_path)
    plt.close() 

@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
        X, Y = get_batch(val_data, context_length, batch_size, device)
        logits, loss = model(X, Y)
        losses[k] = loss.item()
    out['val'] = losses.mean().item()
    model.train()
    return out

# ==========================================
# 4. Model Initialization & Training
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

total_params = sum(p.numel() for p in model.parameters())
print(f"🤖 Total model parameters: {total_params:,}")

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

all_train_losses = [] 
val_losses = []
val_iters = []

checkpoint_dir = 'checkpoints'
os.makedirs(checkpoint_dir, exist_ok=True)
best_val_loss = float('inf')

print("Starting training...")
for iter in tqdm(range(max_iters)):
    
    # ---------------- Evaluation and Checkpoint ----------------
    if iter % eval_interval == 0 or iter == max_iters - 1:
        val_eval = estimate_loss(model)
        val_loss = val_eval['val']
        
        recent_train_loss = sum(all_train_losses[-eval_interval:]) / eval_interval if iter > 0 else 0
        print(f"\nStep {iter}: recent train loss {recent_train_loss:.4f}, val loss {val_loss:.4f}")
        
        val_losses.append(val_loss)
        val_iters.append(iter)

        if len(all_train_losses) > 0:
            save_loss_plot(all_train_losses, val_iters, val_losses)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pt')
            torch.save(model.state_dict(), checkpoint_path)
            print(f"🌟 New best val loss! Saved to {checkpoint_path}")

    # ---------------- Forward & Backward Pass ----------------
    xb, yb = get_batch(train_data, context_length, batch_size, device)
    
    # =========================================================================
    # TODO: Implement the training step.
    # 1. Forward pass: compute `logits` and `loss` using the model, `xb`, and `yb`.
    # 2. Zero the gradients using `optimizer.zero_grad(set_to_none=True)`.
    # 3. Backward pass: compute gradients using `loss.backward()`.
    # 4. Update parameters: step the optimizer.
    # =========================================================================

    loss=None # REPLACE THIS WITH YOUR CODE
    
    #  Append the current loss value (`loss.item()`) to `all_train_losses` list.
    all_train_losses.append(loss.item())

    

save_loss_plot(all_train_losses, val_iters, val_losses)
final_model_path = os.path.join(checkpoint_dir, 'final_model.pt')
torch.save(model.state_dict(), final_model_path)
print(f"✅ Training finished. Final model saved to {final_model_path}")