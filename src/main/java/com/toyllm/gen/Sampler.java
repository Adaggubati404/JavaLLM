package com.toyllm.gen;

import com.toyllm.CharTokenizer;
import com.toyllm.Tensor;
import com.toyllm.model.GPT;

import java.util.Arrays;
import java.util.Random;

/** Autoregressive sampling with optional temperature. */
public final class Sampler {
    private final GPT model;
    private final CharTokenizer tokenizer;
    private final Random rng;

    public Sampler(GPT model, CharTokenizer tokenizer, Random rng) {
        this.model = model;
        this.tokenizer = tokenizer;
        this.rng = rng;
    }

    /**
     * Continue {@code prompt} for {@code maxNewTokens} steps.
     * Breakpoint on logits / probs to compare greedy vs temperature sampling.
     */
    public String generate(String prompt, int maxNewTokens, double temperature) {
        int[] ids = tokenizer.encode(prompt);
        StringBuilder out = new StringBuilder(prompt);

        for (int n = 0; n < maxNewTokens; n++) {
            int[] ctx = ids;
            if (ids.length > model.maxSeqLen) {
                ctx = Arrays.copyOfRange(ids, ids.length - model.maxSeqLen, ids.length);
            }
            Tensor logits = model.forward(ctx);
            int last = ctx.length - 1;
            int v = model.vocabSize;
            double[] row = new double[v];
            System.arraycopy(logits.data, last * v, row, 0, v);

            int nextId;
            if (temperature <= 0.0) {
                nextId = argmax(row);
            } else {
                nextId = sample(row, temperature);
            }
            out.append(tokenizer.decodeOne(nextId));
            ids = Arrays.copyOf(ids, ids.length + 1);
            ids[ids.length - 1] = nextId;
        }
        return out.toString();
    }

    private int argmax(double[] logits) {
        int best = 0;
        for (int i = 1; i < logits.length; i++) {
            if (logits[i] > logits[best]) {
                best = i;
            }
        }
        return best;
    }

    private int sample(double[] logits, double temperature) {
        double max = Double.NEGATIVE_INFINITY;
        for (double logit : logits) {
            max = Math.max(max, logit / temperature);
        }
        double sum = 0.0;
        double[] probs = new double[logits.length];
        for (int i = 0; i < logits.length; i++) {
            probs[i] = Math.exp(logits[i] / temperature - max);
            sum += probs[i];
        }
        double r = rng.nextDouble() * sum;
        double cum = 0.0;
        for (int i = 0; i < probs.length; i++) {
            cum += probs[i];
            if (r <= cum) {
                return i;
            }
        }
        return probs.length - 1;
    }
}
