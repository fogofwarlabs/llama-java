/*
 * Copyright 2026 Fog of War Labs
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.fogofwarlabs.llama;

import static org.fogofwarlabs.llama.bindings.llama_h.C_CHAR;
import static org.fogofwarlabs.llama.bindings.llama_h.C_INT;
import static org.fogofwarlabs.llama.bindings.llama_h.LLAMA_TOKEN_NULL;
import static org.fogofwarlabs.llama.bindings.llama_h.llama_detokenize;
import static org.fogofwarlabs.llama.bindings.llama_h.llama_token_to_piece;
import static org.fogofwarlabs.llama.bindings.llama_h.llama_tokenize;
import static org.fogofwarlabs.llama.bindings.llama_h.llama_vocab_bos;
import static org.fogofwarlabs.llama.bindings.llama_h.llama_vocab_eos;
import static org.fogofwarlabs.llama.bindings.llama_h.llama_vocab_eot;
import static org.fogofwarlabs.llama.bindings.llama_h.llama_vocab_get_add_bos;
import static org.fogofwarlabs.llama.bindings.llama_h.llama_vocab_is_control;
import static org.fogofwarlabs.llama.bindings.llama_h.llama_vocab_is_eog;
import static org.fogofwarlabs.llama.bindings.llama_h.llama_vocab_n_tokens;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.charset.StandardCharsets;
import org.jetbrains.annotations.NotNull;

/**
 * The vocabulary of a {@link LlamaModel}: tokenization, detokenization and the special tokens. It is valid while its model is open, and
 * its methods may be called from several threads.
 */
public final class LlamaVocabulary {
    /**
     * The token value that stands for no token, for example a special token that the vocabulary lacks.
     */
    public static final int NO_TOKEN = LLAMA_TOKEN_NULL();

    private static final int TOKEN_RESERVE = 8;
    private static final int BYTES_PER_TOKEN = 8;
    private static final int PIECE_CAPACITY = 64;

    private final @NotNull LlamaModel model;
    private final @NotNull MemorySegment handle;

    /**
     * Creates a {@link LlamaVocabulary} instance for the {@code llama_vocab} of a model.
     *
     * @param model  the model of the vocabulary
     * @param handle the native vocabulary
     */
    LlamaVocabulary(@NotNull LlamaModel model, @NotNull MemorySegment handle) {
        this.model  = model;
        this.handle = handle;
    }

    /**
     * @return the number of tokens of the vocabulary
     */
    public int getTokenCount() {
        this.model.checkOpen();
        return llama_vocab_n_tokens(this.handle);
    }

    /**
     * @return the beginning-of-sequence token, or {@link #NO_TOKEN}
     */
    public int getBeginToken() {
        this.model.checkOpen();
        return llama_vocab_bos(this.handle);
    }

    /**
     * @return the end-of-sequence token, or {@link #NO_TOKEN}
     */
    public int getEndToken() {
        this.model.checkOpen();
        return llama_vocab_eos(this.handle);
    }

    /**
     * @return the end-of-turn token, or {@link #NO_TOKEN}
     */
    public int getEndOfTurnToken() {
        this.model.checkOpen();
        return llama_vocab_eot(this.handle);
    }

    /**
     * @return {@code true} if tokenization with special tokens adds the beginning-of-sequence token, {@code false} otherwise
     */
    public boolean isBeginTokenAdded() {
        this.model.checkOpen();
        return llama_vocab_get_add_bos(this.handle);
    }

    /**
     * Checks whether a token ends the generation, for example the end-of-sequence or end-of-turn token.
     *
     * @param token the token
     * @return {@code true} if the token ends the generation, {@code false} otherwise
     */
    public boolean isEndOfGeneration(int token) {
        this.model.checkOpen();
        return llama_vocab_is_eog(this.handle, token);
    }

    /**
     * Checks whether a token is a control token.
     *
     * @param token the token
     * @return {@code true} if the token is a control token, {@code false} otherwise
     */
    public boolean isControl(int token) {
        this.model.checkOpen();
        return llama_vocab_is_control(this.handle, token);
    }

    /**
     * Converts a text into tokens.
     *
     * @param text         the text
     * @param addSpecial   whether the beginning-of-sequence and end-of-sequence tokens are added where the model asks for them
     * @param parseSpecial whether special tokens written in the text, for example {@code <|im_start|>}, become those tokens instead of
     *                     plain text
     * @return the tokens
     * @throws LlamaException if the text cannot be tokenized
     */
    public int @NotNull [] tokenize(@NotNull String text, boolean addSpecial, boolean parseSpecial) {
        this.model.checkOpen();

        byte[] bytes = text.getBytes(StandardCharsets.UTF_8);

        try (Arena arena = Arena.ofConfined()) {
            MemorySegment input = arena.allocate(bytes.length + 1L);
            int capacity = bytes.length + TOKEN_RESERVE;
            MemorySegment tokens = arena.allocate(C_INT, capacity);

            MemorySegment.copy(bytes, 0, input, ValueLayout.JAVA_BYTE, 0, bytes.length);

            int count = llama_tokenize(this.handle, input, bytes.length, tokens, capacity, addSpecial, parseSpecial);

            if (count < 0 && count != Integer.MIN_VALUE) {
                capacity = -count;
                tokens   = arena.allocate(C_INT, capacity);
                count    = llama_tokenize(this.handle, input, bytes.length, tokens, capacity, addSpecial, parseSpecial);
            }

            if (count < 0) {
                throw new LlamaException("Cannot tokenize a text of " + bytes.length + " bytes.");
            }

            return tokens.asSlice(0, count * C_INT.byteSize()).toArray(C_INT);
        }
    }

    /**
     * Converts tokens into a text.
     *
     * @param tokens         the tokens
     * @param removeSpecial  whether the beginning-of-sequence and end-of-sequence tokens are removed where the model adds them
     * @param unparseSpecial whether special tokens are written as their text, for example {@code <|im_end|>}, instead of being left out
     * @return the text
     * @throws LlamaException if the tokens cannot be converted
     */
    public @NotNull String detokenize(int @NotNull [] tokens, boolean removeSpecial, boolean unparseSpecial) {
        this.model.checkOpen();

        if (tokens.length == 0) {
            return "";
        }

        try (Arena arena = Arena.ofConfined()) {
            MemorySegment input = arena.allocateFrom(C_INT, tokens);
            int capacity = tokens.length * BYTES_PER_TOKEN + TOKEN_RESERVE;
            MemorySegment text = arena.allocate(capacity);
            int length = llama_detokenize(this.handle, input, tokens.length, text, capacity, removeSpecial, unparseSpecial);

            if (length < 0 && length != Integer.MIN_VALUE) {
                capacity = -length;
                text     = arena.allocate(capacity);
                length   = llama_detokenize(this.handle, input, tokens.length, text, capacity, removeSpecial, unparseSpecial);
            }

            if (length < 0) {
                throw new LlamaException("Cannot detokenize " + tokens.length + " tokens.");
            }

            return new String(text.asSlice(0, length).toArray(C_CHAR), StandardCharsets.UTF_8);
        }
    }

    /**
     * Converts one token into its piece of text. A piece may hold an incomplete UTF-8 sequence: the text of several tokens comes from
     * {@link #detokenize(int[], boolean, boolean)}.
     *
     * @param token   the token
     * @param special whether a special token is written as its text instead of being left out
     * @return the bytes of the piece in UTF-8
     * @throws LlamaException if the token cannot be converted
     */
    public byte @NotNull [] getTokenPiece(int token, boolean special) {
        this.model.checkOpen();

        try (Arena arena = Arena.ofConfined()) {
            int capacity = PIECE_CAPACITY;
            MemorySegment piece = arena.allocate(capacity);
            int length = llama_token_to_piece(this.handle, token, piece, capacity, 0, special);

            if (length < 0) {
                capacity = -length;
                piece    = arena.allocate(capacity);
                length   = llama_token_to_piece(this.handle, token, piece, capacity, 0, special);
            }

            if (length < 0) {
                throw new LlamaException("Cannot convert the token " + token + ".");
            }

            return piece.asSlice(0, length).toArray(C_CHAR);
        }
    }
}
