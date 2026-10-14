# JavaLLM: A Micro-GPT in Pure Java

A tiny, **character-level GPT written in pure Java with no ML libraries** (no native code and no autograd framework). Every piece is hand-written: tensors, forward and backward passes, the Adam optimizer, and sampling. It is built so you can step through a transformer line by line in a JVM debugger.

> This is a learning project, not a production LLM.

## Why I built it

I come from many years of Java engineering, so the JVM debugger is where I understand systems best. To learn how a language model works from first principles, following Andrej Karpathy-style material, I wrote one where nothing is hidden. There is no `loss.backward()` doing the work for me: every gradient is computed by code I wrote. I later rebuilt the ideas at full GPT-2 scale in PyTorch ([TorchLLM](https://github.com/Adaggubati404/TorchLLM)).

## Features

- **Own tensor class** (`Tensor.java`): a row-major N-dimensional array that stores both `data` and `grad`, so you can inspect values and gradients in the Variables pane.
- **Character-level tokenizer:** each distinct character in the corpus is one token.
- **Transformer components written by hand, each with a manual backward pass:**
  - token embedding
  - fixed sinusoidal positional encoding
  - causal multi-head self-attention
  - LayerNorm
  - feed-forward network with GELU (tanh approximation)
  - residual connections
  - LM head
- **Training:** cross-entropy loss, Adam optimizer, and random fixed-length windows sampled from the corpus.
- **Generation:** autoregressive sampling with temperature (greedy when temperature ≤ 0).
- **Checkpointing:** saves and loads model weights to `model.bin`.
- **Built for debugging:**
  - ready-made VS Code / Cursor launch configurations
  - logs of the token embeddings and positional encoding during training
  - an attention heatmap printed at the end of training

## Architecture

```
characters ──► CharTokenizer ──► token embedding + sinusoidal positional encoding
           ──► TransformerBlock × 2 [ LayerNorm → causal multi-head attention → residual
                                      LayerNorm → feed-forward (GELU) → residual ]
           ──► LayerNorm ──► LM head ──► logits ──► cross-entropy loss / sampling
```

| Hyperparameter | Value |
| --- | --- |
| Context length `T` | 32 |
| Model dimension `d` | 32 |
| Attention heads | 2 |
| Layers | 2 |
| Batch size | 8 |
| Optimizer | Adam, lr = 1e-3 |
| Training steps | 100 (`TRAIN_STEPS` in `Main.java`) |

The model trains on a short story in `src/main/resources/corpus.txt`. All hyperparameters are constants in `Main.java`.

## Getting started

**Requirements:**

- **JDK 25** (the project compiles with `maven.compiler.release` 25)
- **Maven 3.9+**

```bash
# Train, save model.bin, and print a sample
mvn -q exec:java -Dexec.args="train"

# Generate from a prompt (needs model.bin from a training run)
mvn -q exec:java -Dexec.args="generate Once upon"
```

During training the program prints the loss at step 1, every 50 steps, and at the final step. At those steps it also prints token-embedding and positional-encoding statistics. When training ends it prints an attention heatmap, saves `model.bin`, and generates a sample from the prompt `Once upon`.

If you change `corpus.txt`, retrain before generating. The checkpoint's vocabulary must match the tokenizer.

### Training output

> **TODO (Ashok):** paste the real loss lines from `mvn -q exec:java -Dexec.args="train"`.

```text
step    1  loss=<value>
step   50  loss=<value>
step  100  loss=<value>
```

### Sample generation

> **TODO (Ashok):** paste the real `--- sample ---` text printed after training.

```text
<paste generated text here>
```

## Debug walkthrough (VS Code / Cursor)

1. Install the **Extension Pack for Java** (recommended in `.vscode/extensions.json`).
2. Open this folder as the workspace so the Maven project `java-toy-llm` is imported.
3. Point the Java settings at your own JDK 25 installation.
4. In **Run and Debug**, pick a configuration:
   - **Debug Train:** train with breakpoints
   - **Debug Generate:** load `model.bin` and sample
   - **Debug Main (no args):** prints usage
   - **Attach to JVM:** attach to a process started with `-agentlib:jdwp=transport=dt_socket,server=y,suspend=y,address=5005`

Suggested breakpoints:

1. `CharTokenizer.encode` / `decode`: characters ↔ ids
2. `PositionalEncoding.forward`: compare PE row 0 with row 1
3. Right after `tokenEmbed + PE` in `GPT.forward`: see how position changes the input
4. `MultiHeadAttention.forward`: watch `lastAttentionWeights` (heads × T × T)
5. `CrossEntropyLoss.forward`: watch `lastProbs`
6. `Sampler.generate`: compare logits for greedy vs. temperature sampling

## Project structure

```
src/main/java/com/toyllm/
  Main.java                  # CLI (train / generate) and hyperparameters
  Tensor.java                # N-d array with data + grad, math ops, GELU
  CharTokenizer.java         # character-level tokenizer
  model/
    Embedding.java
    PositionalEncoding.java  # fixed sin/cos table
    LayerNorm.java
    MultiHeadAttention.java  # causal attention with manual backward
    FeedForward.java
    TransformerBlock.java
    GPT.java                 # full model, save / load
  train/
    CrossEntropyLoss.java
    Adam.java
    Trainer.java             # batching, forward / backward / step, debug dumps
  gen/
    Sampler.java             # greedy and temperature sampling
src/main/resources/corpus.txt
pom.xml
```

## What I learned

- What a tokenizer, an embedding and a positional encoding actually produce, and how position changes the input to the first block.
- How causal masking works inside attention: future positions get negative infinity before the softmax, and their gradients are zeroed in the backward pass.
- Backpropagation without a framework: deriving and coding gradients for attention, LayerNorm, GELU, the residual paths and cross-entropy, and scaling them by the batch size.
- How Adam updates parameters, and how token embeddings move during training while a sinusoidal positional encoding stays fixed.
- How sampling temperature changes generation compared with greedy decoding.

## Next steps

These are not implemented yet:

- Gradient-check tests that compare the manual backward passes with numerical gradients.
- Configurable hyperparameters from the command line instead of constants.
- A larger corpus and longer training, with loss curves.
- Top-k sampling, matching the PyTorch version.

## License

MIT. See [LICENSE](LICENSE).
