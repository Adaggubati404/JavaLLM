package com.toyllm.model;

import com.toyllm.Tensor;

import java.util.List;
import java.util.Random;

/** Position-wise feed-forward: Linear -> GELU -> Linear (dim -> 4*dim -> dim). */
public final class FeedForward {
    public final Tensor w1; // (d, 4d)
    public final Tensor b1; // (4d,)
    public final Tensor w2; // (4d, d)
    public final Tensor b2; // (d,)

    private Tensor lastX;
    private Tensor lastH;      // after w1+b1
    private Tensor lastActivated; // after gelu

    public FeedForward(int dim, Random rng) {
        int hidden = 4 * dim;
        double scale = 0.02;
        this.w1 = Tensor.randn(rng, scale, dim, hidden);
        this.b1 = Tensor.zerosRequiresGrad(hidden);
        this.w2 = Tensor.randn(rng, scale, hidden, dim);
        this.b2 = Tensor.zerosRequiresGrad(dim);
    }

    public Tensor forward(Tensor x) {
        lastX = x;
        int t = x.shape[0];
        int hidden = w1.shape[1];
        int dim = w2.shape[1];

        lastH = Tensor.matmul(x, w1);
        for (int i = 0; i < t; i++) {
            for (int j = 0; j < hidden; j++) {
                lastH.data[i * hidden + j] += b1.data[j];
            }
        }
        lastActivated = Tensor.gelu(lastH);
        Tensor out = Tensor.matmul(lastActivated, w2);
        for (int i = 0; i < t; i++) {
            for (int j = 0; j < dim; j++) {
                out.data[i * dim + j] += b2.data[j];
            }
        }
        return out;
    }

    public void backward(Tensor gradOut, Tensor gradX) {
        int t = lastX.shape[0];
        int dim = lastX.shape[1];
        int hidden = w1.shape[1];

        // db2, dW2, dActivated
        for (int i = 0; i < t; i++) {
            for (int j = 0; j < dim; j++) {
                b2.grad[j] += gradOut.data[i * dim + j];
            }
        }
        Tensor gradActivated = new Tensor(lastActivated.shape, false);
        for (int i = 0; i < t; i++) {
            for (int j = 0; j < hidden; j++) {
                double sum = 0.0;
                for (int k = 0; k < dim; k++) {
                    sum += gradOut.data[i * dim + k] * w2.data[j * dim + k];
                }
                gradActivated.data[i * hidden + j] = sum;
            }
        }
        Tensor.matmulBackwardB(lastActivated, w2, gradOut);

        Tensor gradH = new Tensor(lastH.shape, false);
        Tensor.geluBackward(lastH, gradActivated, gradH);

        for (int i = 0; i < t; i++) {
            for (int j = 0; j < hidden; j++) {
                b1.grad[j] += gradH.data[i * hidden + j];
            }
        }
        // dX = gradH @ W1^T ; dW1 = X^T @ gradH
        for (int i = 0; i < t; i++) {
            for (int j = 0; j < dim; j++) {
                double sum = 0.0;
                for (int k = 0; k < hidden; k++) {
                    sum += gradH.data[i * hidden + k] * w1.data[j * hidden + k];
                }
                gradX.data[i * dim + j] += sum;
            }
        }
        Tensor.matmulBackwardB(lastX, w1, gradH);
        if (lastX.grad != null) {
            lastX.zeroGrad();
        }
    }

    public void parameters(List<Tensor> out) {
        out.add(w1);
        out.add(b1);
        out.add(w2);
        out.add(b2);
    }
}
