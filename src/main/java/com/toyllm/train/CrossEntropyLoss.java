package com.toyllm.train;

import com.toyllm.Tensor;

/**
 * Mean cross-entropy over sequence positions.
 * Softmax is fused for numerical stability; breakpoint on {@link #lastProbs} to see distributions.
 */
public final class CrossEntropyLoss {
    /** Softmax probabilities from the last forward: (T, V). Useful in the debugger. */
    public Tensor lastProbs;

    private Tensor lastLogits;
    private int[] lastTargets;

    /**
     * @param logits  (T, V)
     * @param targets length T token ids
     * @return mean negative log-likelihood
     */
    public double forward(Tensor logits, int[] targets) {
        if (logits.ndim() != 2 || logits.shape[0] != targets.length) {
            throw new IllegalArgumentException("logits/targets shape mismatch");
        }
        lastLogits = logits;
        lastTargets = targets;
        int t = targets.length;
        int v = logits.shape[1];
        lastProbs = new Tensor(logits.shape, false);

        double loss = 0.0;
        for (int i = 0; i < t; i++) {
            int off = i * v;
            double max = logits.data[off];
            for (int j = 1; j < v; j++) {
                max = Math.max(max, logits.data[off + j]);
            }
            double sum = 0.0;
            for (int j = 0; j < v; j++) {
                double e = Math.exp(logits.data[off + j] - max);
                lastProbs.data[off + j] = e;
                sum += e;
            }
            for (int j = 0; j < v; j++) {
                lastProbs.data[off + j] /= sum;
            }
            int y = targets[i];
            double p = Math.max(lastProbs.data[off + y], 1e-12);
            loss += -Math.log(p);
        }
        return loss / t;
    }

    /** Gradient of mean CE w.r.t. logits: (probs - onehot) / T */
    public Tensor backward() {
        int t = lastTargets.length;
        int v = lastLogits.shape[1];
        Tensor grad = new Tensor(lastLogits.shape, false);
        for (int i = 0; i < t; i++) {
            int off = i * v;
            for (int j = 0; j < v; j++) {
                grad.data[off + j] = lastProbs.data[off + j] / t;
            }
            grad.data[off + lastTargets[i]] -= 1.0 / t;
        }
        return grad;
    }
}
