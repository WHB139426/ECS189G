# ECS 189G Spring 2026 - Assignment 1: Training and Probing a Tiny Transformer

In this assignment, you will build, train, and evaluate a minimal decoder-only Transformer from scratch using PyTorch. We will train this model on the Tiny Shakespeare dataset to generate Shakespeare-like text.

## File Structure

- `dataset.py`: Handles downloading the Tiny Shakespeare dataset, character-level tokenization, and batch generation. (Do not modify)
- `model.py`: Contains the core Transformer architecture (Attention, FeedForward, Blocks). **[TODOs inside]**
- `train.py`: The main training script that optimizes the model and plots the loss curves. **[TODOs inside]**
- `inference.py`: Evaluates the model by generating text conditionally based on provided prompts. (Do not modify)
- `requirements.txt`: List of Python packages required for this assignment.

## Environment Setup

It is highly recommended to use a virtual environment (e.g., Conda) to avoid dependency conflicts. 

1. Ensure you have Python 3.8+ installed.
2. Install the required dependencies:
   ```bash
   git clone git@github.com:WHB139426/ECS189G.git
   cd /your/path/to/ECS189G/HW1
   conda create -n tiny_transformer python=3.10.11
   conda activate tiny_transformer
   pip install -r requirements.txt
   ```
(Note: The default code is designed to be CPU-friendly and will automatically use a GPU if available. You do not need a GPU to complete the base requirements of this assignment)

## Task Instructions

### Task 1: Complete the Transformer Model

Open `model.py` and locate the `TODO` blocks. You need to implement:
1. **Scaled Dot-Product Attention**: In the `Head` class, implement the complete attention mechanism:
   - Compute the scaled attention scores by multiplying queries ($Q$) and keys ($K^T$), scaled by $\frac{1}{\sqrt{d_k}}$.
   - Apply the causal mask using `self.tril` to prevent tokens from attending to future tokens.
   - Compute the final output by multiplying the attention weights with the values ($V$).
2. **Cross-Entropy Loss**: In the `TransformerLanguageModel` class, compute the loss between the predicted logits and the target tokens.
3. Run the file directly to test your forward pass and loss computation
   ```bash
   python model.py
   ```
If your implementation is correct, it should print "✅ Forward pass successful!" and output an initial loss value.

### Task 2: Complete the Training Loop
Open `train.py` and locate the `TODO` block inside the training loop. You need to implement the standard PyTorch training step: 
1. Perform the forward pass.
2. Zero the gradients.
3. Perform the backward pass.
4. Step the optimizer.

#### Start Training
   ```bash
   python train.py
   ```
If you correctly fill the `TODO` block, this script will train the model for 5000 iterations. It will automatically save the best model weights to `checkpoints/best_model.pt` and dynamically generate a `loss_curve.png` plot.

### Task 3: Inference and Qualitative Evaluation
Once training is complete, run the inference script to compare the generated text of an untrained model versus your newly trained model.
   ```bash
   python inference.py
   ```
Observe how the trained model learns the structure, character names, and basic vocabulary of Shakespearean plays. 

Note: Due to the extremely small model capacity, limited dataset size, and short training duration, the trained model will likely not output perfectly fluent Shakespearean plays, and you will certainly encounter grammatical and spelling errors. However, compared to the completely random output of the untrained baseline model, the trained model's generated text should demonstrate discernible learned structures and formatting rules (e.g., character names, line breaks, and basic English word formations).

### Task 4: Ablation Studies & Analysis
Modify the hyperparameters at the top of `train.py` to conduct the following experiments:
1. Effect of Context Length: Change `context_length` (e.g., from 64 to 256). Observe and report the impact on the loss curve, training speed, and text generation quality.
2. Effect of Model Size: Increase the model capacity parameters (`n_embd` from 128 to 256, `n_head` from 3 to 6, `n_layer` from 3 to 6, `dropout` from 0 to 0.1). Discuss the trade-offs between parameter count, convergence speed, and potential overfitting.

## Submission Guidelines
Please submit a single .zip file containing:
1. Your completed `model.py` and `train.py`.
2. A PDF Report containing your training loss curve (`loss_curve.png`), generated text samples (output of `inference.py`), and your analysis of the ablation studies (context length and model size).