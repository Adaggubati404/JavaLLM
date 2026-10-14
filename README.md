# Java Toy LLM (Educational Micro-GPT)

A tiny character-level GPT written in **pure Java** (no native ML libs) so you can step through tokenization, sinusoidal positional encoding, attention, loss, and sampling in a JVM debugger.

## Requirements

- JDK 25
- Maven 3.9+

## Run

```bash
# Train (~500 steps) and save model.bin
mvn -q exec:java -Dexec.args="train"

# Generate from a prompt (needs model.bin)
mvn -q exec:java -Dexec.args="generate Once upon"
```

## Model size (intentionally tiny)

| Hyperparam   | Value |
|-------------|-------|
| Context `T` | 32    |
| Dim `d`     | 32    |
| Heads       | 2     |
| Layers      | 2     |
| Batch       | 8     |
| Optimizer   | Adam, lr=1e-3 |

## Architecture

`token embedding + sinusoidal PE → TransformerBlock × 2 → LayerNorm → LM head`

Positional encoding uses the classic sin/cos table (`PositionalEncoding.java`). Open `table` in the debugger to see position/dimension patterns.

## Debug walkthrough (Cursor / VS Code)

1. Install the recommended **Extension Pack for Java** (prompted via `.vscode/extensions.json`).
2. Open this folder as the workspace so Maven project `java-toy-llm` is imported.
3. Open **Run and Debug**, then pick a config:
   - **Debug Train** — train with breakpoints
   - **Debug Generate** — load `model.bin` and sample
   - **Debug Main (no args)** — prints usage
   - **Attach to JVM** — attach to a process started with `-agentlib:jdwp=transport=dt_socket,server=y,suspend=y,address=5005`

JDK 25 is configured in `.vscode/settings.json` (`java.jdt.ls.java.home`).

Suggested breakpoints:

1. `CharTokenizer.encode` / `decode` — confirm characters ↔ ids
2. `PositionalEncoding.forward` — compare PE row 0 vs row 1
3. Right after `tokenEmbed + PE` in `GPT.forward` — see how position changes the input
4. `MultiHeadAttention.forward` — watch `lastAttentionWeights` (heads × T × T)
5. `CrossEntropyLoss.forward` — watch `lastProbs`
6. `Sampler.generate` — compare logits before greedy vs temperature sampling

Variables pane tips:

- Parameter tensors expose both `data` and `grad`
- Attention heatmap is also printed at the end of training

## Project layout

```
src/main/java/com/toyllm/
  Main.java
  Tensor.java
  CharTokenizer.java
  model/   Embedding, PositionalEncoding, LayerNorm, MHA, FFN, Block, GPT
  train/   CrossEntropyLoss, Adam, Trainer
  gen/     Sampler
src/main/resources/corpus.txt
```

This is a learning toy — not a production LLM.
