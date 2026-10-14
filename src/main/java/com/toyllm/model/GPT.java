package com.toyllm.model;

import com.toyllm.Tensor;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * Tiny GPT: token embed + sinusoidal PE + N transformer blocks + final LN + LM head.
 */
public final class GPT {
    public final int vocabSize;
    public final int maxSeqLen;
    public final int dim;
    public final int nLayers;
    public final int nHeads;

    public final Embedding tokenEmbed;
    public final PositionalEncoding posEnc;
    public final TransformerBlock[] blocks;
    public final LayerNorm lnFinal;
    public final Tensor lmHead; // (d, V)

    // caches for backward
    private int[] lastIds;
    private Tensor lastEmbedded; // token + PE
    private Tensor[] lastBlockOut;
    private Tensor lastLn;

    public GPT(int vocabSize, int maxSeqLen, int dim, int nLayers, int nHeads, Random rng) {
        this.vocabSize = vocabSize;
        this.maxSeqLen = maxSeqLen;
        this.dim = dim;
        this.nLayers = nLayers;
        this.nHeads = nHeads;

        this.tokenEmbed = new Embedding(vocabSize, dim, rng);
        this.posEnc = new PositionalEncoding(maxSeqLen, dim);
        this.blocks = new TransformerBlock[nLayers];
        for (int i = 0; i < nLayers; i++) {
            blocks[i] = new TransformerBlock(dim, nHeads, rng);
        }
        this.lnFinal = new LayerNorm(dim);
        this.lmHead = Tensor.randn(rng, 0.02, dim, vocabSize);
    }

    /**
     * Forward: returns logits (T, V).
     * Breakpoint after embed+PE to compare token vs position contributions.
     */
    public Tensor forward(int[] ids) {
        if (ids.length > maxSeqLen) {
            throw new IllegalArgumentException("sequence longer than maxSeqLen");
        }
        lastIds = ids;
        Tensor tok = tokenEmbed.forward(ids);
        Tensor pe = posEnc.forward(ids.length);
        lastEmbedded = Tensor.add(tok, pe);

        Tensor x = lastEmbedded;
        lastBlockOut = new Tensor[nLayers];
        for (int i = 0; i < nLayers; i++) {
            x = blocks[i].forward(x);
            lastBlockOut[i] = x;
        }
        lastLn = lnFinal.forward(x);
        return Tensor.matmul(lastLn, lmHead);
    }

    /**
     * Backprop from logits grad (T, V) into all parameters.
     */
    public void backward(Tensor gradLogits) {
        int t = lastIds.length;

        // logits = ln @ lmHead
        Tensor gradLn = new Tensor(lastLn.shape, false);
        for (int i = 0; i < t; i++) {
            for (int j = 0; j < dim; j++) {
                double sum = 0.0;
                for (int k = 0; k < vocabSize; k++) {
                    sum += gradLogits.data[i * vocabSize + k] * lmHead.data[j * vocabSize + k];
                }
                gradLn.data[i * dim + j] = sum;
            }
        }
        Tensor.matmulBackwardB(lastLn, lmHead, gradLogits);

        Tensor gradX = new Tensor(new int[]{t, dim}, false);
        lnFinal.backward(gradLn, gradX);

        for (int i = nLayers - 1; i >= 0; i--) {
            Tensor gradIn = new Tensor(new int[]{t, dim}, false);
            blocks[i].backward(gradX, gradIn);
            gradX = gradIn;
        }

        // gradX is dL/d(tokenEmbed+PE); PE has no params -> all to token embed
        tokenEmbed.backward(lastIds, gradX);
    }

    public List<Tensor> parameters() {
        List<Tensor> params = new ArrayList<>();
        tokenEmbed.parameters(params);
        for (TransformerBlock b : blocks) {
            b.parameters(params);
        }
        lnFinal.parameters(params);
        params.add(lmHead);
        return params;
    }

    public void zeroGrad() {
        for (Tensor p : parameters()) {
            p.zeroGrad();
        }
    }

    public void save(Path path) throws IOException {
        try (DataOutputStream out = new DataOutputStream(Files.newOutputStream(path))) {
            out.writeInt(vocabSize);
            out.writeInt(maxSeqLen);
            out.writeInt(dim);
            out.writeInt(nLayers);
            out.writeInt(nHeads);
            for (Tensor p : parameters()) {
                out.writeInt(p.data.length);
                for (double v : p.data) {
                    out.writeDouble(v);
                }
            }
        }
    }

    public static GPT load(Path path, Random rng) throws IOException {
        try (DataInputStream in = new DataInputStream(Files.newInputStream(path))) {
            int vocabSize = in.readInt();
            int maxSeqLen = in.readInt();
            int dim = in.readInt();
            int nLayers = in.readInt();
            int nHeads = in.readInt();
            GPT model = new GPT(vocabSize, maxSeqLen, dim, nLayers, nHeads, rng);
            for (Tensor p : model.parameters()) {
                int len = in.readInt();
                if (len != p.data.length) {
                    throw new IOException("checkpoint size mismatch: " + len + " vs " + p.data.length);
                }
                for (int i = 0; i < len; i++) {
                    p.data[i] = in.readDouble();
                }
            }
            return model;
        }
    }
}
