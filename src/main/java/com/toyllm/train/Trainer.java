package com.toyllm.train;

import com.toyllm.CharTokenizer;
import com.toyllm.Tensor;
import com.toyllm.model.GPT;

import java.util.Random;

/**
 * Samples fixed-length windows from the corpus and runs forward / backward / Adam steps.
 */
public final class Trainer {
    private final GPT model;
    private final CharTokenizer tokenizer;
    private final int[] allIds;
    private final int contextLen;
    private final int batchSize;
    private final Adam adam;
    private final CrossEntropyLoss lossFn = new CrossEntropyLoss();
    private final Random rng;
    /** Snapshot of token-embed weights after the last dumpEmbedAndPe call. */
    private Tensor prevEmbedSnapshot;

    public Trainer(GPT model, CharTokenizer tokenizer, String corpus,
                   int contextLen, int batchSize, double lr, Random rng) {
        this.model = model;
        this.tokenizer = tokenizer;
        this.allIds = tokenizer.encode(corpus);
        this.contextLen = contextLen;
        this.batchSize = batchSize;
        this.adam = new Adam(lr);
        this.rng = rng;
        if (allIds.length < contextLen + 1) {
            throw new IllegalArgumentException("corpus too short for contextLen=" + contextLen);
        }
    }

    public CrossEntropyLoss lossFn() {
        return lossFn;
    }

    /**
     * One optimization step over {@code batchSize} random windows.
     * @return mean loss over the batch
     */
    public double step() {
        model.zeroGrad();
        double totalLoss = 0.0;

        for (int b = 0; b < batchSize; b++) {
            int start = rng.nextInt(allIds.length - contextLen);
            int[] x = new int[contextLen];
            int[] y = new int[contextLen];
            System.arraycopy(allIds, start, x, 0, contextLen);
            System.arraycopy(allIds, start + 1, y, 0, contextLen);

            Tensor logits = model.forward(x);
            double loss = lossFn.forward(logits, y);
            totalLoss += loss;

            Tensor gradLogits = lossFn.backward();
            // average over batch: scale grads by 1/batchSize
            for (int i = 0; i < gradLogits.data.length; i++) {
                gradLogits.data[i] /= batchSize;
            }
            model.backward(gradLogits);
        }

        adam.step(model.parameters());
        return totalLoss / batchSize;
    }

    /**
     * Log token-embedding evolution vs fixed sinusoidal PE.
     * PE values stay identical every call; embed L2 / ΔL2 should move with Adam.
     */
    public void dumpEmbedAndPe(int step) {
        Tensor pe = model.posEnc.table;
        Tensor emb = model.tokenEmbed.weight;
        int d = emb.shape[1];

        System.out.printf("--- embed/PE @ step %d ---%n", step);
        System.out.printf("PE (fixed sin/cos): L2=%.4f mean=%.4f  [unchanged across steps]%n",
                pe.l2(), pe.mean());
        System.out.printf("  pos0[:4]=%s%n", formatHead(pe, 0, d, 4));
        System.out.printf("  pos1[:4]=%s%n", formatHead(pe, 1, d, 4));

        String deltaStr = prevEmbedSnapshot == null
                ? "n/a"
                : String.format("%.6f", emb.l2Diff(prevEmbedSnapshot));
        System.out.printf("Embed (V x d): L2=%.4f mean=%.4f std=%.4f  ΔL2_vs_prev=%s%n",
                emb.l2(), emb.mean(), emb.std(), deltaStr);

        char[] sampleChars = {'e', 'a', 't', ' '};
        Integer scaleTokId = null;
        for (char c : sampleChars) {
            int id = tokenizer.idOf(c);
            if (id < 0) {
                continue;
            }
            if (scaleTokId == null) {
                scaleTokId = id;
            }
            double rowL2 = rowL2(emb, id, d);
            System.out.printf("  '%s' id=%d rowL2=%.4f [:4]=%s%n",
                    escapeChar(c), id, rowL2, formatHead(emb, id, d, 4));
        }

        if (scaleTokId != null) {
            double tokNorm = rowL2(emb, scaleTokId, d);
            double pe0Norm = rowL2(pe, 0, d);
            char label = tokenizer.decodeOne(scaleTokId).charAt(0);
            System.out.printf("tok vs PE scale @ pos0: ||tok('%s')|| / ||PE[0]|| = %.4f%n",
                    escapeChar(label), tokNorm / pe0Norm);
        }

        prevEmbedSnapshot = emb.copyData();
    }

    /** Print a small attention heatmap from the last forward of block 0. */
    public void dumpAttention() {
        System.out.println(model.blocks[0].attn.attentionHeatmap(0));
    }

    private static double rowL2(Tensor t, int row, int d) {
        double sumSq = 0.0;
        int off = row * d;
        for (int j = 0; j < d; j++) {
            double v = t.data[off + j];
            sumSq += v * v;
        }
        return Math.sqrt(sumSq);
    }

    private static String formatHead(Tensor t, int row, int d, int n) {
        int k = Math.min(n, d);
        StringBuilder sb = new StringBuilder("[");
        int off = row * d;
        for (int j = 0; j < k; j++) {
            if (j > 0) {
                sb.append(", ");
            }
            sb.append(String.format("%.4f", t.data[off + j]));
        }
        sb.append(']');
        return sb.toString();
    }

    private static String escapeChar(char c) {
        if (c == ' ') {
            return " ";
        }
        if (c == '\n') {
            return "\\n";
        }
        if (c == '\t') {
            return "\\t";
        }
        return String.valueOf(c);
    }
}
