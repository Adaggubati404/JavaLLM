package com.toyllm.model;

import com.toyllm.Tensor;

/**
 * Fixed sinusoidal positional encoding (Vaswani et al.).
 * Precomputes a (maxSeq, d) table — open {@link #table} in the debugger to see sin/cos patterns.
 * No trainable parameters; gradients through the add go only to token embeddings.
 */
public final class PositionalEncoding {
    /** Full PE table: shape (maxSeqLen, dim). Inspect in debugger. */
    public final Tensor table;

    public PositionalEncoding(int maxSeqLen, int dim) {
        this.table = Tensor.zeros(maxSeqLen, dim);
        for (int pos = 0; pos < maxSeqLen; pos++) {
            for (int i = 0; i < dim; i++) {
                double angle = pos / Math.pow(10000.0, (2.0 * (i / 2)) / dim);
                if (i % 2 == 0) {
                    table.data[pos * dim + i] = Math.sin(angle);
                } else {
                    table.data[pos * dim + i] = Math.cos(angle);
                }
            }
        }
    }

    /**
     * Returns PE rows [0, seqLen). Breakpoint here and compare position 0 vs position 1.
     */
    public Tensor forward(int seqLen) {
        if (seqLen > table.shape[0]) {
            throw new IllegalArgumentException("seqLen " + seqLen + " > max " + table.shape[0]);
        }
        int d = table.shape[1];
        Tensor out = new Tensor(new int[]{seqLen, d}, false);
        System.arraycopy(table.data, 0, out.data, 0, seqLen * d);
        return out;
    }
}
