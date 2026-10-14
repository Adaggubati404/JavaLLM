package com.toyllm;

import com.toyllm.gen.Sampler;
import com.toyllm.model.GPT;
import com.toyllm.train.Trainer;

import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.Random;

/**
 * CLI entry: {@code train} or {@code generate [prompt...]}.
 *
 * Suggested debugger breakpoints:
 * <ul>
 *   <li>{@link com.toyllm.model.PositionalEncoding#forward}</li>
 *   <li>{@link com.toyllm.model.MultiHeadAttention#forward}</li>
 *   <li>{@link com.toyllm.train.CrossEntropyLoss#forward}</li>
 *   <li>{@link com.toyllm.gen.Sampler#generate}</li>
 * </ul>
 */
public final class Main {
    public static final int CONTEXT_LEN = 32;
    public static final int DIM = 32;
    public static final int N_HEADS = 2;
    public static final int N_LAYERS = 2;
    public static final int BATCH_SIZE = 8;
    public static final double LR = 1e-3;
    public static final int TRAIN_STEPS = 100;
    public static final Path CHECKPOINT = Path.of("model.bin");

    public static void main(String[] args) throws Exception {
        if (args.length == 0) {
            printUsage();
            return;
        }
        String corpus = loadCorpus();
        CharTokenizer tokenizer = new CharTokenizer(corpus);
        Random rng = new Random(42);

        switch (args[0]) {
            case "train" -> train(corpus, tokenizer, rng);
            case "generate" -> generate(args, corpus, tokenizer, rng);
            default -> {
                System.err.println("Unknown command: " + args[0]);
                printUsage();
            }
        }
    }

    private static void train(String corpus, CharTokenizer tokenizer, Random rng) throws Exception {
        System.out.println("Vocab size: " + tokenizer.vocabSize());
        System.out.println("Corpus chars: " + corpus.length());
        System.out.printf("Model: d=%d heads=%d layers=%d context=%d%n",
                DIM, N_HEADS, N_LAYERS, CONTEXT_LEN);

        GPT model = new GPT(tokenizer.vocabSize(), CONTEXT_LEN, DIM, N_LAYERS, N_HEADS, rng);
        Trainer trainer = new Trainer(model, tokenizer, corpus, CONTEXT_LEN, BATCH_SIZE, LR, rng);

        for (int step = 1; step <= TRAIN_STEPS; step++) {
            double loss = trainer.step();
            if (step == 1 || step % 50 == 0 || step == TRAIN_STEPS) {
                System.out.printf("step %4d  loss=%.4f%n", step, loss);
                trainer.dumpEmbedAndPe(step);
            }
            if (step == TRAIN_STEPS) {
                trainer.dumpAttention();
            }
        }

        model.save(CHECKPOINT);
        System.out.println("Saved checkpoint to " + CHECKPOINT.toAbsolutePath());

        Sampler sampler = new Sampler(model, tokenizer, rng);
        String sample = sampler.generate("Once upon", 80, 0.8);
        System.out.println("--- sample ---");
        System.out.println(sample);
    }

    private static void generate(String[] args, String corpus, CharTokenizer tokenizer, Random rng)
            throws Exception {
        if (!CHECKPOINT.toFile().exists()) {
            System.err.println("No " + CHECKPOINT + " found. Run `train` first.");
            return;
        }
        GPT model = GPT.load(CHECKPOINT, rng);
        if (model.vocabSize != tokenizer.vocabSize()) {
            System.err.println("Checkpoint vocab (" + model.vocabSize
                    + ") != tokenizer vocab (" + tokenizer.vocabSize()
                    + "). Retrain with the current corpus.");
            return;
        }
        String prompt = args.length > 1
                ? String.join(" ", java.util.Arrays.copyOfRange(args, 1, args.length))
                : "Once upon";
        Sampler sampler = new Sampler(model, tokenizer, rng);
        String text = sampler.generate(prompt, 120, 0.8);
        System.out.println(text);
    }

    private static String loadCorpus() throws Exception {
        try (InputStream in = Main.class.getResourceAsStream("/corpus.txt")) {
            if (in == null) {
                throw new IllegalStateException("corpus.txt not found on classpath");
            }
            return new String(in.readAllBytes(), StandardCharsets.UTF_8);
        }
    }

    private static void printUsage() {
        System.out.println("""
                Java Toy LLM — educational micro-GPT
                
                Usage:
                  mvn -q exec:java -Dexec.args="train"
                  mvn -q exec:java -Dexec.args="generate Once upon"
                
                Debug in Cursor/VS Code with launch configs "Debug Train" / "Debug Generate".
                """);
    }
}
