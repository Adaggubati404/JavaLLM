# JavaLLM: A Micro-GPT in Pure Java

A tiny, **character-level GPT written in pure Java with no ML libraries** (no native code and no autograd framework). Every piece is hand-written: tensors, forward and backward passes, the Adam optimizer, and sampling. It is built so you can step through a transformer line by line in a JVM debugger.

> A learning project for understanding transformer internals, not a production LLM.

## Why I built it

I come from many years of Java engineering, so the JVM debugger is where I understand systems best. To learn how a language model works from first principles, following Andrej Karpathy-style material, I wrote one where nothing is hidden. There is no `loss.backward()` doing the work for me: every gradient is computed by code I wrote. I later rebuilt the ideas at full GPT-2 scale in PyTorch ([TorchLLM](https://github.com/Adaggubati404/TorchLLM)).

## What this taught me

- **How data moves during training.** Characters become token ids, then embeddings plus positional encodings, then pass through the transformer blocks to logits. Every intermediate tensor is a Java array you can inspect.
- **How matrix multiplication works in the model.** The query, key and value projections, the attention scores, the output projection and the LM head are all `Tensor.matmul` calls written by hand.
- **How loss is calculated.** Softmax turns the logits into probabilities, and cross-entropy measures how much probability the model gave the correct next character.
- **How backpropagation drives training.** Every layer has a hand-written backward pass that sends gradients from the loss back through the LM head, LayerNorm, the feed-forward network, attention and the embeddings. Adam then uses those gradients to update the weights.
- **Why inference doesn't need backpropagation.** Generation only runs the forward pass and samples the next token. Gradients are needed only to change the weights, and at inference the weights are fixed.
- **How temperature shapes creative output.** Dividing the logits by the temperature sharpens the distribution (close to greedy) or flattens it (more varied, more "creative" text). Temperature ≤ 0 falls back to greedy decoding.
- **How layer normalization helps training.** Normalizing each position's activations keeps them on a stable scale between blocks, which keeps gradients well-behaved and training steady.

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

> **Note:** This is a learning project built to understand the internals; it is not trained to convergence. The tiny model and corpus are there so every step can be inspected, not to produce fluent text.

## Stepping through it in a debugger

IDE launch configurations are not included in this repo. Open it as a Maven project in any Java IDE (IntelliJ IDEA, VS Code with the Extension Pack for Java, Eclipse), set JDK 25, and run `com.toyllm.Main` with the argument `train` or `generate Once upon`. To debug a Maven run instead, start it with `MAVEN_OPTS="-agentlib:jdwp=transport=dt_socket,server=y,suspend=y,address=5005" mvn -q exec:java -Dexec.args="train"` and attach your IDE's remote debugger to port 5005.

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

## Next steps

These are not implemented yet:

- Gradient-check tests that compare the manual backward passes with numerical gradients.
- Configurable hyperparameters from the command line instead of constants.
- Top-k sampling, matching the PyTorch version.

## Roadmap

This project is one step in learning the AI stack from the ground up before moving into agentic engineering. **Next: fine-tuning the open-source Qwen model and building a real agentic application on top of it.**

## License

MIT. See [LICENSE](LICENSE).
