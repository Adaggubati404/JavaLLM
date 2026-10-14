package com.toyllm;

import java.util.Arrays;
import java.util.Random;
import java.util.concurrent.ThreadLocalRandom;

/**
 * Small educational ND array: row-major {@code data} plus matching {@code grad}
 * so you can inspect both in the debugger Variables pane.
 */
public final class Tensor {
    public final int[] shape;
    public final double[] data;
    public final double[] grad;
    public final boolean requiresGrad;

    public Tensor(int[] shape, boolean requiresGrad) {
        this.shape = shape.clone();
        this.data = new double[numel(shape)];
        this.requiresGrad = requiresGrad;
        this.grad = requiresGrad ? new double[data.length] : null;
    }

    public Tensor(int[] shape, double[] data, boolean requiresGrad) {
        this.shape = shape.clone();
        if (data.length != numel(shape)) {
            throw new IllegalArgumentException("data length " + data.length + " != numel " + numel(shape));
        }
        this.data = data;
        this.requiresGrad = requiresGrad;
        this.grad = requiresGrad ? new double[data.length] : null;
    }

    public static int numel(int[] shape) {
        int n = 1;
        for (int s : shape) {
            n *= s;
        }
        return n;
    }

    public int ndim() {
        return shape.length;
    }

    public int size(int dim) {
        return shape[dim];
    }

    public static Tensor zeros(int... shape) {
        return new Tensor(shape, false);
    }

    public static Tensor zerosRequiresGrad(int... shape) {
        return new Tensor(shape, true);
    }

    public static Tensor randn(Random rng, double scale, int... shape) {
        Tensor t = new Tensor(shape, true);
        for (int i = 0; i < t.data.length; i++) {
            t.data[i] = rng.nextGaussian() * scale;
        }
        return t;
    }

    public static Tensor randn(double scale, int... shape) {
        return randn(ThreadLocalRandom.current(), scale, shape);
    }

    public void zeroGrad() {
        if (grad != null) {
            Arrays.fill(grad, 0.0);
        }
    }

    public Tensor copyData() {
        return new Tensor(shape, Arrays.copyOf(data, data.length), false);
    }

    /** Mean of all data elements (for training logs). */
    public double mean() {
        double sum = 0.0;
        for (double v : data) {
            sum += v;
        }
        return sum / data.length;
    }

    /** Population std of all data elements (for training logs). */
    public double std() {
        double m = mean();
        double sumSq = 0.0;
        for (double v : data) {
            double d = v - m;
            sumSq += d * d;
        }
        return Math.sqrt(sumSq / data.length);
    }

    /** L2 (Frobenius) norm of data. */
    public double l2() {
        double sumSq = 0.0;
        for (double v : data) {
            sumSq += v * v;
        }
        return Math.sqrt(sumSq);
    }

    /** L2 norm of element-wise difference vs {@code other} (same length). */
    public double l2Diff(Tensor other) {
        if (other.data.length != data.length) {
            throw new IllegalArgumentException("l2Diff length mismatch");
        }
        double sumSq = 0.0;
        for (int i = 0; i < data.length; i++) {
            double d = data[i] - other.data[i];
            sumSq += d * d;
        }
        return Math.sqrt(sumSq);
    }

    public int index(int... indices) {
        if (indices.length != shape.length) {
            throw new IllegalArgumentException("index rank mismatch");
        }
        int idx = 0;
        for (int i = 0; i < shape.length; i++) {
            if (indices[i] < 0 || indices[i] >= shape[i]) {
                throw new IndexOutOfBoundsException(Arrays.toString(indices));
            }
            idx = idx * shape[i] + indices[i];
        }
        return idx;
    }

    public double get(int... indices) {
        return data[index(indices)];
    }

    public void set(double value, int... indices) {
        data[index(indices)] = value;
    }

    /** Element-wise add; result does not require grad (caller manages backprop). */
    public static Tensor add(Tensor a, Tensor b) {
        checkSameShape(a, b);
        Tensor out = new Tensor(a.shape, false);
        for (int i = 0; i < out.data.length; i++) {
            out.data[i] = a.data[i] + b.data[i];
        }
        return out;
    }

    public static Tensor addInPlace(Tensor a, Tensor b) {
        checkSameShape(a, b);
        for (int i = 0; i < a.data.length; i++) {
            a.data[i] += b.data[i];
        }
        return a;
    }

    public static Tensor scale(Tensor a, double s) {
        Tensor out = new Tensor(a.shape, false);
        for (int i = 0; i < out.data.length; i++) {
            out.data[i] = a.data[i] * s;
        }
        return out;
    }

    /** Matrix multiply: (m,k) @ (k,n) -> (m,n). */
    public static Tensor matmul(Tensor a, Tensor b) {
        if (a.ndim() != 2 || b.ndim() != 2) {
            throw new IllegalArgumentException("matmul expects 2D tensors");
        }
        int m = a.shape[0];
        int k = a.shape[1];
        int k2 = b.shape[0];
        int n = b.shape[1];
        if (k != k2) {
            throw new IllegalArgumentException("matmul inner dim mismatch: " + k + " vs " + k2);
        }
        Tensor out = new Tensor(new int[]{m, n}, false);
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                double sum = 0.0;
                for (int t = 0; t < k; t++) {
                    sum += a.data[i * k + t] * b.data[t * n + j];
                }
                out.data[i * n + j] = sum;
            }
        }
        return out;
    }

    /** Transpose of a 2D tensor. */
    public static Tensor transpose(Tensor a) {
        if (a.ndim() != 2) {
            throw new IllegalArgumentException("transpose expects 2D");
        }
        int m = a.shape[0];
        int n = a.shape[1];
        Tensor out = new Tensor(new int[]{n, m}, false);
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                out.data[j * m + i] = a.data[i * n + j];
            }
        }
        return out;
    }

    /**
     * Softmax along the last dimension for a 2D tensor (rows independently).
     * Numerically stable (subtract max).
     */
    public static Tensor softmaxRows(Tensor a) {
        if (a.ndim() != 2) {
            throw new IllegalArgumentException("softmaxRows expects 2D");
        }
        int rows = a.shape[0];
        int cols = a.shape[1];
        Tensor out = new Tensor(a.shape, false);
        for (int r = 0; r < rows; r++) {
            int off = r * cols;
            double max = a.data[off];
            for (int c = 1; c < cols; c++) {
                max = Math.max(max, a.data[off + c]);
            }
            double sum = 0.0;
            for (int c = 0; c < cols; c++) {
                double e = Math.exp(a.data[off + c] - max);
                out.data[off + c] = e;
                sum += e;
            }
            for (int c = 0; c < cols; c++) {
                out.data[off + c] /= sum;
            }
        }
        return out;
    }

    /** GELU approximation (tanh form). */
    public static Tensor gelu(Tensor a) {
        Tensor out = new Tensor(a.shape, false);
        for (int i = 0; i < a.data.length; i++) {
            double x = a.data[i];
            out.data[i] = 0.5 * x * (1.0 + Math.tanh(Math.sqrt(2.0 / Math.PI) * (x + 0.044715 * x * x * x)));
        }
        return out;
    }

    /** Derivative of GELU (tanh approx) w.r.t. input, element-wise. */
    public static void geluBackward(Tensor x, Tensor gradOut, Tensor gradIn) {
        for (int i = 0; i < x.data.length; i++) {
            double v = x.data[i];
            double v3 = v * v * v;
            double inner = Math.sqrt(2.0 / Math.PI) * (v + 0.044715 * v3);
            double tanh = Math.tanh(inner);
            double sech2 = 1.0 - tanh * tanh;
            double dInner = Math.sqrt(2.0 / Math.PI) * (1.0 + 3.0 * 0.044715 * v * v);
            double dGelu = 0.5 * (1.0 + tanh) + 0.5 * v * sech2 * dInner;
            gradIn.data[i] += gradOut.data[i] * dGelu;
        }
    }

    /**
     * Softmax-row backward: given y = softmax(x) (rows), accumulate dL/dx into gradX.
     * Formula: dx_i = y_i * (dy_i - sum_j y_j * dy_j)
     */
    public static void softmaxRowsBackward(Tensor y, Tensor gradY, Tensor gradX) {
        int rows = y.shape[0];
        int cols = y.shape[1];
        for (int r = 0; r < rows; r++) {
            int off = r * cols;
            double dot = 0.0;
            for (int c = 0; c < cols; c++) {
                dot += y.data[off + c] * gradY.data[off + c];
            }
            for (int c = 0; c < cols; c++) {
                gradX.data[off + c] += y.data[off + c] * (gradY.data[off + c] - dot);
            }
        }
    }

    /** Accumulate A^T @ B into outGrad for A when out = A @ B (A is (m,k), B is (k,n)). */
    public static void matmulBackwardA(Tensor a, Tensor b, Tensor gradOut) {
        // dA = gradOut @ B^T
        int m = a.shape[0];
        int k = a.shape[1];
        int n = b.shape[1];
        for (int i = 0; i < m; i++) {
            for (int t = 0; t < k; t++) {
                double sum = 0.0;
                for (int j = 0; j < n; j++) {
                    sum += gradOut.data[i * n + j] * b.data[t * n + j];
                }
                a.grad[i * k + t] += sum;
            }
        }
    }

    /** Accumulate A @ B contribution into B.grad when out = A @ B. */
    public static void matmulBackwardB(Tensor a, Tensor b, Tensor gradOut) {
        // dB = A^T @ gradOut
        int m = a.shape[0];
        int k = a.shape[1];
        int n = b.shape[1];
        for (int t = 0; t < k; t++) {
            for (int j = 0; j < n; j++) {
                double sum = 0.0;
                for (int i = 0; i < m; i++) {
                    sum += a.data[i * k + t] * gradOut.data[i * n + j];
                }
                b.grad[t * n + j] += sum;
            }
        }
    }

    /** Write gradOut into gradIn (same shape), for residual / identity paths. */
    public static void accumulateGrad(Tensor target, Tensor gradSrc) {
        checkSameShape(target, gradSrc);
        if (target.grad == null) {
            throw new IllegalStateException("target has no grad buffer");
        }
        for (int i = 0; i < target.grad.length; i++) {
            target.grad[i] += gradSrc.data[i];
        }
    }

    public static void accumulateGradFrom(Tensor targetGradHolder, double[] src) {
        if (targetGradHolder.grad == null) {
            throw new IllegalStateException("no grad");
        }
        if (src.length != targetGradHolder.grad.length) {
            throw new IllegalArgumentException("length mismatch");
        }
        for (int i = 0; i < src.length; i++) {
            targetGradHolder.grad[i] += src[i];
        }
    }

    private static void checkSameShape(Tensor a, Tensor b) {
        if (!Arrays.equals(a.shape, b.shape)) {
            throw new IllegalArgumentException(
                    "shape mismatch: " + Arrays.toString(a.shape) + " vs " + Arrays.toString(b.shape));
        }
    }

    @Override
    public String toString() {
        return "Tensor(shape=" + Arrays.toString(shape)
                + ", requiresGrad=" + requiresGrad
                + ", data[0.." + Math.min(4, data.length) + ")="
                + Arrays.toString(Arrays.copyOf(data, Math.min(4, data.length))) + "...)";
    }
}
