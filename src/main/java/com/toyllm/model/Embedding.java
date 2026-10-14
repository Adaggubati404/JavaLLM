package com.toyllm.model;

import com.toyllm.Tensor;

import java.util.List;
import java.util.Random;

/** Learnable token embedding table: vocabSize x dim. */
public final class Embedding {
    public final Tensor weight; // (V, d)

    public Embedding(int vocabSize, int dim, Random rng) {
        double scale = 0.02;
        this.weight = Tensor.randn(rng, scale, vocabSize, dim);
    }

    /** Lookup rows for each token id. Returns (T, d). */
    public Tensor forward(int[] ids) {
        int t = ids.length;
        int d = weight.shape[1];
        Tensor out = new Tensor(new int[]{t, d}, false);
        for (int i = 0; i < t; i++) {
            int row = ids[i];
            System.arraycopy(weight.data, row * d, out.data, i * d, d);
        }
        return out;
    }

    /** Scatter-add gradients from (T, d) into weight.grad rows. */
    public void backward(int[] ids, Tensor gradOut) {
        int d = weight.shape[1];
        for (int i = 0; i < ids.length; i++) {
            int row = ids[i];
            int src = i * d;
            int dst = row * d;
            for (int j = 0; j < d; j++) {
                weight.grad[dst + j] += gradOut.data[src + j];
            }
        }
    }

    public void parameters(List<Tensor> out) {
        out.add(weight);
    }
}
