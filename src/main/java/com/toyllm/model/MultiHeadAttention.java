package com.toyllm.model;

import com.toyllm.Tensor;

import java.util.List;
import java.util.Random;

/**
 * Causal multi-head self-attention.
 * After forward, inspect {@link #lastAttentionWeights} — shape (heads, T, T) — to see who attends to whom.
 */
public final class MultiHeadAttention {
    public final int nHeads;
    public final int dim;
    public final int headDim;

    public final Tensor wq;
    public final Tensor wk;
    public final Tensor wv;
    public final Tensor wo;

    /** Debugger watch: attention probs after softmax + causal mask. Shape (H, T, T). */
    public Tensor lastAttentionWeights;

    // caches
    private Tensor lastX;
    private Tensor lastQ; // (T, d)
    private Tensor lastK;
    private Tensor lastV;
    private Tensor lastConcat; // after heads concat, before Wo (T, d)
    private Tensor[] lastScores; // per head (T, T) pre-softmax
    private Tensor[] lastAttn;   // per head (T, T) post-softmax
    private Tensor[] lastHeadOut; // per head (T, headDim)

    public MultiHeadAttention(int dim, int nHeads, Random rng) {
        if (dim % nHeads != 0) {
            throw new IllegalArgumentException("dim must be divisible by nHeads");
        }
        this.dim = dim;
        this.nHeads = nHeads;
        this.headDim = dim / nHeads;
        double scale = 0.02;
        this.wq = Tensor.randn(rng, scale, dim, dim);
        this.wk = Tensor.randn(rng, scale, dim, dim);
        this.wv = Tensor.randn(rng, scale, dim, dim);
        this.wo = Tensor.randn(rng, scale, dim, dim);
    }

    public Tensor forward(Tensor x) {
        // x: (T, d)
        lastX = x;
        int t = x.shape[0];

        lastQ = Tensor.matmul(x, wq);
        lastK = Tensor.matmul(x, wk);
        lastV = Tensor.matmul(x, wv);

        lastScores = new Tensor[nHeads];
        lastAttn = new Tensor[nHeads];
        lastHeadOut = new Tensor[nHeads];
        lastAttentionWeights = new Tensor(new int[]{nHeads, t, t}, false);

        lastConcat = new Tensor(new int[]{t, dim}, false);
        double scale = 1.0 / Math.sqrt(headDim);

        for (int h = 0; h < nHeads; h++) {
            Tensor qh = sliceHead(lastQ, h, t);
            Tensor kh = sliceHead(lastK, h, t);
            Tensor vh = sliceHead(lastV, h, t);

            // scores = qh @ kh^T * scale, causal mask
            Tensor scores = new Tensor(new int[]{t, t}, false);
            for (int i = 0; i < t; i++) {
                for (int j = 0; j < t; j++) {
                    if (j > i) {
                        scores.data[i * t + j] = Double.NEGATIVE_INFINITY;
                    } else {
                        double sum = 0.0;
                        for (int k = 0; k < headDim; k++) {
                            sum += qh.data[i * headDim + k] * kh.data[j * headDim + k];
                        }
                        scores.data[i * t + j] = sum * scale;
                    }
                }
            }
            lastScores[h] = scores;
            Tensor attn = Tensor.softmaxRows(scores);
            // fix NaN from -inf rows (shouldn't happen for causal with j<=i)
            for (int i = 0; i < attn.data.length; i++) {
                if (Double.isNaN(attn.data[i])) {
                    attn.data[i] = 0.0;
                }
            }
            lastAttn[h] = attn;
            System.arraycopy(attn.data, 0, lastAttentionWeights.data, h * t * t, t * t);

            // out = attn @ vh
            Tensor hout = new Tensor(new int[]{t, headDim}, false);
            for (int i = 0; i < t; i++) {
                for (int k = 0; k < headDim; k++) {
                    double sum = 0.0;
                    for (int j = 0; j < t; j++) {
                        sum += attn.data[i * t + j] * vh.data[j * headDim + k];
                    }
                    hout.data[i * headDim + k] = sum;
                }
            }
            lastHeadOut[h] = hout;
            writeHead(lastConcat, h, t, hout);
        }

        return Tensor.matmul(lastConcat, wo);
    }

    public void backward(Tensor gradOut, Tensor gradX) {
        int t = lastX.shape[0];

        // grad through Wo: out = concat @ wo
        Tensor gradConcat = new Tensor(lastConcat.shape, false);
        // d(concat) = gradOut @ wo^T ; d(wo) = concat^T @ gradOut
        for (int i = 0; i < t; i++) {
            for (int j = 0; j < dim; j++) {
                double sum = 0.0;
                for (int k = 0; k < dim; k++) {
                    sum += gradOut.data[i * dim + k] * wo.data[j * dim + k];
                }
                gradConcat.data[i * dim + j] = sum;
            }
        }
        Tensor.matmulBackwardB(lastConcat, wo, gradOut);

        Tensor gradQ = new Tensor(lastQ.shape, false);
        Tensor gradK = new Tensor(lastK.shape, false);
        Tensor gradV = new Tensor(lastV.shape, false);
        double scale = 1.0 / Math.sqrt(headDim);

        for (int h = 0; h < nHeads; h++) {
            Tensor gradHout = sliceHead(gradConcat, h, t);
            Tensor vh = sliceHead(lastV, h, t);
            Tensor attn = lastAttn[h];
            Tensor qh = sliceHead(lastQ, h, t);
            Tensor kh = sliceHead(lastK, h, t);

            // out = attn @ vh  => dAttn = gradHout @ vh^T ; dVh = attn^T @ gradHout
            Tensor gradAttn = new Tensor(new int[]{t, t}, false);
            Tensor gradVh = new Tensor(new int[]{t, headDim}, false);
            for (int i = 0; i < t; i++) {
                for (int j = 0; j < t; j++) {
                    double sum = 0.0;
                    for (int k = 0; k < headDim; k++) {
                        sum += gradHout.data[i * headDim + k] * vh.data[j * headDim + k];
                    }
                    gradAttn.data[i * t + j] = sum;
                }
            }
            for (int j = 0; j < t; j++) {
                for (int k = 0; k < headDim; k++) {
                    double sum = 0.0;
                    for (int i = 0; i < t; i++) {
                        sum += attn.data[i * t + j] * gradHout.data[i * headDim + k];
                    }
                    gradVh.data[j * headDim + k] = sum;
                }
            }
            writeHead(gradV, h, t, gradVh);

            // softmax backward -> gradScores
            Tensor gradScores = new Tensor(new int[]{t, t}, false);
            Tensor.softmaxRowsBackward(attn, gradAttn, gradScores);
            // causal: positions j>i were -inf; zero their grad
            for (int i = 0; i < t; i++) {
                for (int j = i + 1; j < t; j++) {
                    gradScores.data[i * t + j] = 0.0;
                }
            }

            // scores = scale * qh @ kh^T
            // dQh = scale * gradScores @ kh ; dKh = scale * gradScores^T @ qh
            Tensor gradQh = new Tensor(new int[]{t, headDim}, false);
            Tensor gradKh = new Tensor(new int[]{t, headDim}, false);
            for (int i = 0; i < t; i++) {
                for (int k = 0; k < headDim; k++) {
                    double sumQ = 0.0;
                    double sumK = 0.0;
                    for (int j = 0; j < t; j++) {
                        sumQ += gradScores.data[i * t + j] * kh.data[j * headDim + k] * scale;
                        sumK += gradScores.data[j * t + i] * qh.data[j * headDim + k] * scale;
                    }
                    gradQh.data[i * headDim + k] = sumQ;
                    gradKh.data[i * headDim + k] = sumK;
                }
            }
            writeHead(gradQ, h, t, gradQh);
            writeHead(gradK, h, t, gradKh);
        }

        // Weight grads: dW = X^T @ grad(Q/K/V). Activation lastX has no grad buffer.
        Tensor.matmulBackwardB(lastX, wq, gradQ);
        Tensor.matmulBackwardB(lastX, wk, gradK);
        Tensor.matmulBackwardB(lastX, wv, gradV);

        // dX = gradQ @ Wq^T + gradK @ Wk^T + gradV @ Wv^T
        for (int i = 0; i < t; i++) {
            for (int j = 0; j < dim; j++) {
                double sum = 0.0;
                for (int k = 0; k < dim; k++) {
                    sum += gradQ.data[i * dim + k] * wq.data[j * dim + k];
                    sum += gradK.data[i * dim + k] * wk.data[j * dim + k];
                    sum += gradV.data[i * dim + k] * wv.data[j * dim + k];
                }
                gradX.data[i * dim + j] += sum;
            }
        }
    }

    private Tensor sliceHead(Tensor full, int head, int t) {
        Tensor out = new Tensor(new int[]{t, headDim}, false);
        for (int i = 0; i < t; i++) {
            System.arraycopy(full.data, i * dim + head * headDim, out.data, i * headDim, headDim);
        }
        return out;
    }

    private void writeHead(Tensor full, int head, int t, Tensor headTensor) {
        for (int i = 0; i < t; i++) {
            System.arraycopy(headTensor.data, i * headDim, full.data, i * dim + head * headDim, headDim);
        }
    }

    public void parameters(List<Tensor> out) {
        out.add(wq);
        out.add(wk);
        out.add(wv);
        out.add(wo);
    }

    /** ASCII heatmap of attention for head 0 (for training logs). */
    public String attentionHeatmap(int head) {
        if (lastAttentionWeights == null) {
            return "(no attention yet)";
        }
        int t = lastAttentionWeights.shape[1];
        StringBuilder sb = new StringBuilder();
        sb.append("Attention head ").append(head).append(" (").append(t).append("x").append(t).append("):\n");
        int off = head * t * t;
        for (int i = 0; i < t; i++) {
            for (int j = 0; j < t; j++) {
                double v = lastAttentionWeights.data[off + i * t + j];
                if (j > i) {
                    sb.append(' ');
                } else if (v < 0.05) {
                    sb.append('.');
                } else if (v < 0.15) {
                    sb.append('o');
                } else if (v < 0.35) {
                    sb.append('O');
                } else {
                    sb.append('#');
                }
            }
            sb.append('\n');
        }
        return sb.toString();
    }
}
