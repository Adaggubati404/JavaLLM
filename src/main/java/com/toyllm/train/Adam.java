package com.toyllm.train;

import com.toyllm.Tensor;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

/** Adam optimizer over a list of parameter tensors. */
public final class Adam {
    private final double lr;
    private final double beta1;
    private final double beta2;
    private final double eps;
    private final Map<Tensor, double[]> m = new HashMap<>();
    private final Map<Tensor, double[]> v = new HashMap<>();
    private int step;

    public Adam(double lr) {
        this(lr, 0.9, 0.999, 1e-8);
    }

    public Adam(double lr, double beta1, double beta2, double eps) {
        this.lr = lr;
        this.beta1 = beta1;
        this.beta2 = beta2;
        this.eps = eps;
    }

    public void step(List<Tensor> params) {
        step++;
        double bc1 = 1.0 - Math.pow(beta1, step);
        double bc2 = 1.0 - Math.pow(beta2, step);
        for (Tensor p : params) {
            if (p.grad == null) {
                continue;
            }
            double[] mm = m.computeIfAbsent(p, t -> new double[t.data.length]);
            double[] vv = v.computeIfAbsent(p, t -> new double[t.data.length]);
            for (int i = 0; i < p.data.length; i++) {
                double g = p.grad[i];
                mm[i] = beta1 * mm[i] + (1.0 - beta1) * g;
                vv[i] = beta2 * vv[i] + (1.0 - beta2) * g * g;
                double mHat = mm[i] / bc1;
                double vHat = vv[i] / bc2;
                p.data[i] -= lr * mHat / (Math.sqrt(vHat) + eps);
            }
        }
    }
}
