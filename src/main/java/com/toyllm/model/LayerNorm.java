package com.toyllm.model;

import com.toyllm.Tensor;

import java.util.Arrays;
import java.util.List;

/** Layer normalization over the last dimension (features). */
public final class LayerNorm {
    public final Tensor gamma; // (d,)
    public final Tensor beta;  // (d,)
    private final double eps;

    // caches for backward
    private Tensor lastX;
    private Tensor lastNormed; // (x - mean) / sqrt(var+eps)
    private double[] lastInvStd; // per row
    private double[] lastMean;

    public LayerNorm(int dim) {
        this.eps = 1e-5;
        this.gamma = Tensor.zerosRequiresGrad(dim);
        this.beta = Tensor.zerosRequiresGrad(dim);
        Arrays.fill(gamma.data, 1.0);
    }

    public Tensor forward(Tensor x) {
        // x: (T, d)
        int t = x.shape[0];
        int d = x.shape[1];
        lastX = x;
        lastNormed = new Tensor(x.shape, false);
        lastInvStd = new double[t];
        lastMean = new double[t];
        Tensor out = new Tensor(x.shape, false);

        for (int i = 0; i < t; i++) {
            int off = i * d;
            double mean = 0.0;
            for (int j = 0; j < d; j++) {
                mean += x.data[off + j];
            }
            mean /= d;
            lastMean[i] = mean;

            double var = 0.0;
            for (int j = 0; j < d; j++) {
                double diff = x.data[off + j] - mean;
                var += diff * diff;
            }
            var /= d;
            double invStd = 1.0 / Math.sqrt(var + eps);
            lastInvStd[i] = invStd;

            for (int j = 0; j < d; j++) {
                double normed = (x.data[off + j] - mean) * invStd;
                lastNormed.data[off + j] = normed;
                out.data[off + j] = normed * gamma.data[j] + beta.data[j];
            }
        }
        return out;
    }

    /**
     * @param gradOut dL/dy shape (T, d)
     * @param gradX   accumulator for dL/dx (same shape); may be a plain Tensor with .data used as grad buffer
     */
    public void backward(Tensor gradOut, Tensor gradX) {
        int t = lastX.shape[0];
        int d = lastX.shape[1];

        for (int i = 0; i < t; i++) {
            int off = i * d;
            // dgamma, dbeta
            for (int j = 0; j < d; j++) {
                gamma.grad[j] += gradOut.data[off + j] * lastNormed.data[off + j];
                beta.grad[j] += gradOut.data[off + j];
            }

            // dx via standard LayerNorm backward
            double invStd = lastInvStd[i];
            double[] dyHat = new double[d];
            for (int j = 0; j < d; j++) {
                dyHat[j] = gradOut.data[off + j] * gamma.data[j];
            }
            double sumDyHat = 0.0;
            double sumDyHatNormed = 0.0;
            for (int j = 0; j < d; j++) {
                sumDyHat += dyHat[j];
                sumDyHatNormed += dyHat[j] * lastNormed.data[off + j];
            }
            for (int j = 0; j < d; j++) {
                double dx = (dyHat[j] - sumDyHat / d - lastNormed.data[off + j] * sumDyHatNormed / d) * invStd;
                gradX.data[off + j] += dx;
            }
        }
    }

    public void parameters(List<Tensor> out) {
        out.add(gamma);
        out.add(beta);
    }
}
