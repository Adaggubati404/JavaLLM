package com.toyllm.model;

import com.toyllm.Tensor;

import java.util.List;
import java.util.Random;

/** Pre-norm transformer block: x + attn(ln1(x)); x + ffn(ln2(x)). */
public final class TransformerBlock {
    public final LayerNorm ln1;
    public final MultiHeadAttention attn;
    public final LayerNorm ln2;
    public final FeedForward ffn;

    private Tensor lastX;
    private Tensor lastLn1;
    private Tensor lastAttnOut;
    private Tensor lastRes1;
    private Tensor lastLn2;
    private Tensor lastFfnOut;

    public TransformerBlock(int dim, int nHeads, Random rng) {
        this.ln1 = new LayerNorm(dim);
        this.attn = new MultiHeadAttention(dim, nHeads, rng);
        this.ln2 = new LayerNorm(dim);
        this.ffn = new FeedForward(dim, rng);
    }

    public Tensor forward(Tensor x) {
        lastX = x;
        lastLn1 = ln1.forward(x);
        lastAttnOut = attn.forward(lastLn1);
        lastRes1 = Tensor.add(x, lastAttnOut);

        lastLn2 = ln2.forward(lastRes1);
        lastFfnOut = ffn.forward(lastLn2);
        return Tensor.add(lastRes1, lastFfnOut);
    }

    public void backward(Tensor gradOut, Tensor gradX) {
        // out = res1 + ffn  => d(res1) += gradOut, d(ffn) = gradOut
        Tensor gradRes1 = gradOut.copyData();
        Tensor gradFfn = gradOut.copyData();

        Tensor gradLn2 = new Tensor(lastLn2.shape, false);
        ffn.backward(gradFfn, gradLn2);
        ln2.backward(gradLn2, gradRes1);

        // res1 = x + attnOut
        Tensor gradAttn = gradRes1.copyData();
        // also residual to x
        for (int i = 0; i < gradX.data.length; i++) {
            gradX.data[i] += gradRes1.data[i];
        }

        Tensor gradLn1 = new Tensor(lastLn1.shape, false);
        attn.backward(gradAttn, gradLn1);
        ln1.backward(gradLn1, gradX);
    }

    public void parameters(List<Tensor> out) {
        ln1.parameters(out);
        attn.parameters(out);
        ln2.parameters(out);
        ffn.parameters(out);
    }
}
