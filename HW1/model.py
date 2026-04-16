import torch
import torch.nn as nn
from torch.nn import functional as F

class Head(nn.Module):
    def __init__(self, head_size, n_embd, context_length, dropout):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(context_length, context_length)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)   # (B, T, head_size)
        q = self.query(x) # (B, T, head_size)
        
        # =========================================================================
        # TODO 1: Compute the scaled attention scores ('wei').
        # Multiply the query (q) and key (k) tensors, and scale the result by 
        # the square root of the channel dimension (C ** -0.5).
        # Hint: You will need to transpose the last two dimensions of k.
        # =========================================================================
        
        wei = None # REPLACE THIS LINE
        
        # =========================================================================
        # TODO 2: Implement the causal mask for the attention mechanism.
        # Use `self.tril` to mask out future tokens in the 'wei' tensor.
        # Fill the positions where the mask is 0 with float('-inf').
        # =========================================================================
        
        wei = None # REPLACE THIS LINE
        
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        
        v = self.value(x) # (B, T, head_size)
        
        # =========================================================================
        # TODO 3: Compute the final output of the attention head.
        # Multiply the attention weights ('wei') by the values ('v').
        # =========================================================================
        
        out = None # REPLACE THIS LINE
        
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size, n_embd, context_length, dropout):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size, n_embd, context_length, dropout) for _ in range(num_heads)])
        self.proj = nn.Linear(num_heads * head_size, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Module):
    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embd, n_head, context_length, dropout):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size, n_embd, context_length, dropout)
        self.ffwd = FeedFoward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))   
        x = x + self.ffwd(self.ln2(x)) 
        return x

class TransformerLanguageModel(nn.Module):
    def __init__(self, vocab_size, context_length, n_embd, n_head, n_layer, dropout, device):
        super().__init__()
        self.context_length = context_length
        self.device = device
        
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(context_length, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head, context_length, dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) 
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        
        tok_emb = self.token_embedding_table(idx) 
        pos_emb = self.position_embedding_table(torch.arange(T, device=self.device)) 
        x = tok_emb + pos_emb 
        
        x = self.blocks(x) 
        x = self.ln_f(x)
        logits = self.lm_head(x) 

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            
            # =========================================================================
            # TODO: Calculate the cross-entropy loss between logits and targets.
            # Hint: Use F.cross_entropy
            # =========================================================================
            
            loss = None # REPLACE THIS LINE

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.context_length:] 
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :] 
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1) 
        return idx

if __name__ == '__main__':
    # =========================================================================
    # Quick Local Test
    # Run `python model.py` to check if your model implementation is runnable.
    # =========================================================================
    print("--- Running Quick Model Sanity Check ---")
    
    # Dummy hyperparameters
    vocab_size = 65
    context_length = 8
    n_embd = 32
    n_head = 2
    n_layer = 2
    dropout = 0.0
    device = 'cpu'
    
    # Initialize the model
    model = TransformerLanguageModel(
        vocab_size, context_length, n_embd, n_head, n_layer, dropout, device
    )
    
    # Dummy inputs (Batch size: 4, Sequence length: 8)
    dummy_idx = torch.randint(0, vocab_size, (4, context_length))
    dummy_targets = torch.randint(0, vocab_size, (4, context_length))
    
    try:
        logits, loss = model(dummy_idx, dummy_targets)
        print(f"✅ Forward pass successful!")
        print(f"👉 Logits shape: {list(logits.shape)} (Expected: [32, 65])")
        
        if loss is not None:
            print(f"👉 Initial Loss value: {loss.item():.4f}")
        else:
            print("⚠️ Loss is None. Make sure you have implemented the loss TODO!")
    except Exception as e:
        print(f"❌ Error during forward pass: {e}")
    
    print("----------------------------------------")