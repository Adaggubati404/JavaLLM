package com.toyllm;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.TreeSet;

/**
 * Character-level tokenizer: each distinct character in the corpus is one token.
 * Easy to inspect in the debugger: encode("ab") -> [id_a, id_b].
 */
public final class CharTokenizer {
    private final Map<Character, Integer> charToId = new HashMap<>();
    private final List<Character> idToChar = new ArrayList<>();

    public CharTokenizer(String text) {
        TreeSet<Character> chars = new TreeSet<>();
        for (int i = 0; i < text.length(); i++) {
            chars.add(text.charAt(i));
        }
        for (char c : chars) {
            charToId.put(c, idToChar.size());
            idToChar.add(c);
        }
    }

    public int vocabSize() {
        return idToChar.size();
    }

    public int[] encode(String text) {
        int[] ids = new int[text.length()];
        for (int i = 0; i < text.length(); i++) {
            char c = text.charAt(i);
            Integer id = charToId.get(c);
            if (id == null) {
                // Unknown char: map to first vocab id (toy behavior)
                id = 0;
            }
            ids[i] = id;
        }
        return ids;
    }

    public String decode(int[] ids) {
        StringBuilder sb = new StringBuilder(ids.length);
        for (int id : ids) {
            if (id >= 0 && id < idToChar.size()) {
                sb.append(idToChar.get(id));
            }
        }
        return sb.toString();
    }

    public String decodeOne(int id) {
        if (id < 0 || id >= idToChar.size()) {
            return "?";
        }
        return String.valueOf(idToChar.get(id));
    }

    /** Returns vocab id, or -1 if {@code c} is not in the vocabulary. */
    public int idOf(char c) {
        Integer id = charToId.get(c);
        return id == null ? -1 : id;
    }
}
