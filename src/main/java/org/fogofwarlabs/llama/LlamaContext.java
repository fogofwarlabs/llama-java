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

import static org.fogofwarlabs.llama.bindings.llama_h.C_INT;
import static org.fogofwarlabs.llama.bindings.llama_h.LLAMA_PROCESS_TYPE_DECODE;
import static org.fogofwarlabs.llama.bindings.llama_h.llama_batch_ext_add_token;
import static org.fogofwarlabs.llama.bindings.llama_h.llama_batch_ext_clear;
import static org.fogofwarlabs.llama.bindings.llama_h.llama_batch_ext_free;
import static org.fogofwarlabs.llama.bindings.llama_h.llama_batch_ext_init;
import static org.fogofwarlabs.llama.bindings.llama_h.llama_batch_ext_set_output_logits;
import static org.fogofwarlabs.llama.bindings.llama_h.llama_batch_ext_set_pos;
import static org.fogofwarlabs.llama.bindings.llama_h.llama_context_default_params;
import static org.fogofwarlabs.llama.bindings.llama_h.llama_free;
import static org.fogofwarlabs.llama.bindings.llama_h.llama_get_memory;
import static org.fogofwarlabs.llama.bindings.llama_h.llama_init_from_model;
import static org.fogofwarlabs.llama.bindings.llama_h.llama_memory_clear;
import static org.fogofwarlabs.llama.bindings.llama_h.llama_n_batch;
import static org.fogofwarlabs.llama.bindings.llama_h.llama_n_ctx;
import static org.fogofwarlabs.llama.bindings.llama_h.llama_n_ctx_seq;
import static org.fogofwarlabs.llama.bindings.llama_h.llama_n_seq_max;
import static org.fogofwarlabs.llama.bindings.llama_h.llama_n_ubatch;
import static org.fogofwarlabs.llama.bindings.llama_h.llama_process;
import static org.fogofwarlabs.llama.bindings.llama_h.llama_sampler_chain_add;
import static org.fogofwarlabs.llama.bindings.llama_h.llama_sampler_chain_default_params;
import static org.fogofwarlabs.llama.bindings.llama_h.llama_sampler_chain_init;
import static org.fogofwarlabs.llama.bindings.llama_h.llama_sampler_free;
import static org.fogofwarlabs.llama.bindings.llama_h.llama_sampler_init_greedy;
import static org.fogofwarlabs.llama.bindings.llama_h.llama_sampler_sample;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.time.Duration;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import org.fogofwarlabs.llama.bindings.llama_context_params;
import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;

/**
 * A context of a {@link LlamaModel}: the memory of the sequences and the greedy generation of texts from prompts.
 * <p>
 * Every request starts with an empty memory. The prompts are tokenized with special tokens parsed, so a prompt in a chat format holds
 * its control tokens as text, for example {@code <|im_start|>}. Several prompts of one request are decoded as parallel sequences in the
 * same batches. A sequence stops at an end-of-generation token of the vocabulary, at the maximum number of tokens or when its part of the
 * context memory is full.
 * <p>
 * The methods of a context are synchronized; a context decodes one request at a time.
 */
public final class LlamaContext implements AutoCloseable {
    private static final int BATCH_FULL = -1;

    private static final int PROCESS_SUCCESS = 0;
    private static final int PROCESS_NO_MEMORY_SLOT = 1;
    private static final int PROCESS_ABORTED = 2;

    /**
     * The prompt sizes of {@link #warmUp()}: GPU backends choose their matrix kernels by the number of tokens of a batch.
     */
    private static final int @NotNull [] WARM_UP_PROMPT_SIZES = {16, 48, 96};

    /**
     * The state of one sequence of a request: its prompt, the generated tokens and the position of the next token.
     */
    private static final class Sequence {
        private final int id;
        private final int @NotNull [] promptTokens;

        private final int @NotNull [] tokens;
        private int tokenCount;
        private int position;
        private int outputIndex;
        private @Nullable StopReason stopReason;
        private long stopTime;

        private Sequence(int id, int @NotNull [] promptTokens, int maximumTokens) {
            this.id           = id;
            this.promptTokens = promptTokens;

            this.tokens      = new int[maximumTokens];
            this.tokenCount  = 0;
            this.position    = promptTokens.length;
            this.outputIndex = BATCH_FULL;
            this.stopReason  = null;
            this.stopTime    = 0L;
        }

        private boolean isActive() {
            return this.stopReason == null;
        }

        private int getLastToken() {
            return this.tokens[this.tokenCount - 1];
        }

        /**
         * Adds a sampled token, or stops the sequence at an end-of-generation token, at the maximum number of tokens or when the next
         * position lies outside its part of the context memory.
         */
        private void accept(int token, @NotNull LlamaVocabulary vocabulary, int contextSize, long time) {
            if (vocabulary.isEndOfGeneration(token)) {
                this.stop(StopReason.END_OF_GENERATION, time);
                return;
            }

            this.tokens[this.tokenCount++] = token;

            if (this.tokenCount == this.tokens.length) {
                this.stop(StopReason.MAXIMUM_TOKENS, time);
            } else if (this.position >= contextSize) {
                this.stop(StopReason.CONTEXT_FULL, time);
            }
        }

        private void stop(@NotNull StopReason reason, long time) {
            this.stopReason = reason;
            this.stopTime   = time;
        }

        private @NotNull Generation createGeneration(@NotNull LlamaVocabulary vocabulary, long startTime, long promptEndTime) {
            int[] generated = Arrays.copyOf(this.tokens, this.tokenCount);
            String text = vocabulary.detokenize(generated, false, false);
            StopReason reason = this.stopReason == null ? StopReason.MAXIMUM_TOKENS : this.stopReason;
            // A sequence that stops at the token of its prompt stops before the prompt time ends.
            long generationTime = Math.max(this.stopTime - promptEndTime, 0L);

            return new Generation(text, generated, reason,
                                  this.promptTokens.length, Duration.ofNanos(promptEndTime - startTime), Duration.ofNanos(generationTime));
        }
    }

    private final @NotNull LlamaModel model;
    private final @NotNull LlamaVocabulary vocabulary;

    private final @NotNull Arena arena;
    private final @NotNull MemorySegment handle;
    private final @NotNull MemorySegment batch;
    private final @NotNull MemorySegment sampler;
    private final @NotNull MemorySegment memory;
    private final @NotNull MemorySegment position;

    private final int contextSize;
    private final int sequenceContextSize;
    private final int batchSize;
    private final int microBatchSize;
    private final int sequenceCount;

    private boolean closed;

    /**
     * Creates a {@link LlamaContext} instance on a model.
     *
     * @param model   the model
     * @param options the sizes and threads of the context
     * @throws LlamaException if the context cannot be created
     */
    LlamaContext(@NotNull LlamaModel model, @NotNull ContextOptions options) {
        Arena arena = Arena.ofShared();
        MemorySegment handle = MemorySegment.NULL;
        MemorySegment batch = MemorySegment.NULL;
        MemorySegment sampler;

        try {
            MemorySegment parameters = llama_context_default_params(arena);

            llama_context_params.n_ctx(parameters, options.getContextSize());
            llama_context_params.n_batch(parameters, options.getBatchSize());
            llama_context_params.n_ubatch(parameters, options.getMicroBatchSize());
            llama_context_params.n_seq_max(parameters, options.getSequenceCount());
            llama_context_params.kv_unified(parameters, options.isUnifiedMemory());
            llama_context_params.n_threads(parameters, options.getThreadCount());
            llama_context_params.n_threads_batch(parameters, options.getBatchThreadCount());

            handle = llama_init_from_model(model.getHandle(), parameters);

            if (handle.equals(MemorySegment.NULL)) {
                throw new LlamaException("Cannot create a context of " + options.getContextSize() + " tokens; the log org.fogofwarlabs.llama " +
                                         "names the cause.");
            }

            batch = llama_batch_ext_init(handle);

            if (batch.equals(MemorySegment.NULL)) {
                throw new LlamaException("Cannot create the batch of a context.");
            }

            sampler = llama_sampler_chain_init(llama_sampler_chain_default_params(arena));
            llama_sampler_chain_add(sampler, llama_sampler_init_greedy());
        } catch (RuntimeException exception) {
            if (!batch.equals(MemorySegment.NULL)) {
                llama_batch_ext_free(batch);
            }

            if (!handle.equals(MemorySegment.NULL)) {
                llama_free(handle);
            }

            arena.close();
            throw exception;
        }

        this.model      = model;
        this.vocabulary = model.getVocabulary();

        this.arena    = arena;
        this.handle   = handle;
        this.batch    = batch;
        this.sampler  = sampler;
        this.memory   = llama_get_memory(handle);
        this.position = arena.allocate(C_INT);

        this.contextSize         = llama_n_ctx(handle);
        this.sequenceContextSize = llama_n_ctx_seq(handle);
        this.batchSize           = llama_n_batch(handle);
        this.microBatchSize      = llama_n_ubatch(handle);
        this.sequenceCount       = llama_n_seq_max(handle);

        this.closed = false;
    }

    /**
     * @return the model of this context
     */
    public @NotNull LlamaModel getModel() {
        return this.model;
    }

    /**
     * @return the context size in tokens, shared by all sequences
     */
    public int getContextSize() {
        return this.contextSize;
    }

    /**
     * @return the number of tokens that one sequence can hold: the context size, or its equal part when the memory is not unified
     */
    public int getSequenceContextSize() {
        return this.sequenceContextSize;
    }

    /**
     * @return the number of tokens that one decoding step takes
     */
    public int getBatchSize() {
        return this.batchSize;
    }

    /**
     * @return the number of tokens that the backend computes at once
     */
    public int getMicroBatchSize() {
        return this.microBatchSize;
    }

    /**
     * @return the number of sequences that one request can generate in parallel
     */
    public int getSequenceCount() {
        return this.sequenceCount;
    }

    /**
     * Decodes batches of the shapes that requests use and clears the memory again, so that the backend prepares its kernels before the
     * first request: prompts of a few, some dozens and about a hundred tokens, and steps of one token for one up to all sequences. A GPU
     * backend may take seconds for that.
     *
     * @throws LlamaException if a batch cannot be decoded
     */
    public synchronized void warmUp() {
        this.checkOpen();

        int token = this.vocabulary.getBeginToken();

        if (token == LlamaVocabulary.NO_TOKEN) {
            token = this.vocabulary.getEndToken();
        }

        if (token == LlamaVocabulary.NO_TOKEN) {
            token = 0;
        }

        int maximumPromptSize = Math.min(this.batchSize, this.sequenceContextSize - 1);

        for (int promptSize : WARM_UP_PROMPT_SIZES) {
            int size = Math.min(promptSize, maximumPromptSize);
            int index = BATCH_FULL;

            llama_memory_clear(this.memory, true);
            llama_batch_ext_clear(this.batch);

            for (int position = 0; position < size; position++) {
                index = this.addToken(0, token, position, position == size - 1);
            }

            this.process();
            llama_sampler_sample(this.sampler, this.handle, index);
        }

        // Step n holds one token of each of the first n sequences; sequence s starts at step s + 1.
        llama_memory_clear(this.memory, true);

        for (int step = 1; step <= this.sequenceCount; step++) {
            int index = BATCH_FULL;

            llama_batch_ext_clear(this.batch);

            for (int sequenceId = 0; sequenceId < step; sequenceId++) {
                index = this.addToken(sequenceId, token, step - 1 - sequenceId, true);
            }

            this.process();
            llama_sampler_sample(this.sampler, this.handle, index);
        }

        llama_memory_clear(this.memory, true);
    }

    /**
     * Generates a text from one prompt.
     *
     * @param prompt        the prompt, with the special tokens of the model's chat format written as text
     * @param maximumTokens the maximum number of generated tokens
     * @return the generation
     * @throws IllegalArgumentException if the maximum is not positive or the prompt is empty or does not fit into a sequence
     * @throws LlamaException           if decoding fails
     */
    public @NotNull Generation generate(@NotNull String prompt, int maximumTokens) {
        return this.generate(List.of(prompt), maximumTokens).getFirst();
    }

    /**
     * Generates a text from each prompt; the prompts are decoded as parallel sequences in the same batches.
     *
     * @param prompts       the prompts, at most {@link #getSequenceCount()}, with the special tokens of the model's chat format written
     *                      as text
     * @param maximumTokens the maximum number of generated tokens of each sequence
     * @return the generations, in the order of the prompts
     * @throws IllegalArgumentException if the maximum is not positive, there are more prompts than sequences, or a prompt is empty or does
     *                                  not fit into a sequence
     * @throws LlamaException           if decoding fails, for example when the sequences of a unified memory together fill it
     */
    public synchronized @NotNull List<Generation> generate(@NotNull List<String> prompts, int maximumTokens) {
        this.checkOpen();

        if (maximumTokens <= 0) {
            throw new IllegalArgumentException("Invalid maximum number of tokens: " + maximumTokens);
        }

        if (prompts.size() > this.sequenceCount) {
            throw new IllegalArgumentException(prompts.size() + " prompts exceed the " + this.sequenceCount + " sequences of the context.");
        }

        List<Sequence> sequences = new ArrayList<>(prompts.size());

        for (int index = 0; index < prompts.size(); index++) {
            int[] promptTokens = this.vocabulary.tokenize(prompts.get(index), true, true);

            if (promptTokens.length == 0) {
                throw new IllegalArgumentException("The prompt " + index + " has no tokens.");
            }

            if (promptTokens.length >= this.sequenceContextSize) {
                throw new IllegalArgumentException("The prompt " + index + " has " + promptTokens.length + " tokens; a sequence of the context " +
                                                   "holds " + this.sequenceContextSize + ".");
            }

            sequences.add(new Sequence(index, promptTokens, maximumTokens));
        }

        llama_memory_clear(this.memory, true);

        long startTime = System.nanoTime();

        this.decodePrompts(sequences);

        long promptEndTime = System.nanoTime();

        this.decodeTokens(sequences);

        List<Generation> generations = new ArrayList<>(sequences.size());

        for (Sequence sequence : sequences) {
            generations.add(sequence.createGeneration(this.vocabulary, startTime, promptEndTime));
        }

        return generations;
    }

    /**
     * Decodes the prompts of all sequences and samples the first token of each. A batch holds the tokens of several prompts; a prompt
     * that does not fit continues in the next batch.
     */
    private void decodePrompts(@NotNull List<Sequence> sequences) {
        List<Sequence> sampled = new ArrayList<>(sequences.size());

        llama_batch_ext_clear(this.batch);

        for (Sequence sequence : sequences) {
            int[] tokens = sequence.promptTokens;

            for (int index = 0; index < tokens.length; index++) {
                boolean last = index == tokens.length - 1;
                int batchIndex = this.addToken(sequence.id, tokens[index], index, last);

                if (batchIndex == BATCH_FULL) {
                    this.process();
                    this.sample(sampled);
                    sampled.clear();

                    llama_batch_ext_clear(this.batch);
                    batchIndex = this.addToken(sequence.id, tokens[index], index, last);

                    if (batchIndex == BATCH_FULL) {
                        throw new LlamaException("An empty batch takes no token.");
                    }
                }

                if (last) {
                    sequence.outputIndex = batchIndex;
                    sampled.add(sequence);
                }
            }
        }

        this.process();
        this.sample(sampled);
    }

    /**
     * Decodes the last token of every active sequence in one batch and samples the next tokens, until every sequence stopped.
     */
    private void decodeTokens(@NotNull List<Sequence> sequences) {
        List<Sequence> active = new ArrayList<>(sequences.size());

        while (true) {
            active.clear();

            for (Sequence sequence : sequences) {
                if (sequence.isActive()) {
                    active.add(sequence);
                }
            }

            if (active.isEmpty()) {
                return;
            }

            llama_batch_ext_clear(this.batch);

            for (Sequence sequence : active) {
                sequence.outputIndex = this.addToken(sequence.id, sequence.getLastToken(), sequence.position, true);

                if (sequence.outputIndex == BATCH_FULL) {
                    throw new LlamaException("A batch of " + this.batchSize + " tokens cannot hold " + active.size() + " sequences.");
                }

                sequence.position++;
            }

            this.process();
            this.sample(active);
        }
    }

    private int addToken(int sequenceId, int token, int position, boolean output) {
        int index = llama_batch_ext_add_token(this.batch, sequenceId, token);

        if (index == BATCH_FULL) {
            return BATCH_FULL;
        }

        if (index < 0) {
            throw new LlamaException("Cannot add the token " + token + " of the sequence " + sequenceId + " to the batch: error " + index + ".");
        }

        this.position.set(C_INT, 0, position);
        llama_batch_ext_set_pos(this.batch, index, this.position);

        if (output) {
            llama_batch_ext_set_output_logits(this.batch, index, true);
        }

        return index;
    }

    private void process() {
        int result = llama_process(this.handle, LLAMA_PROCESS_TYPE_DECODE(), this.batch);

        if (result == PROCESS_SUCCESS) {
            return;
        }

        if (result == PROCESS_NO_MEMORY_SLOT) {
            throw new LlamaException("The context memory of " + this.contextSize + " tokens has no room for the batch.");
        }

        if (result == PROCESS_ABORTED) {
            throw new LlamaException("Decoding was aborted.");
        }

        throw new LlamaException("Decoding failed with error " + result + "; the log org.fogofwarlabs.llama names the cause.");
    }

    private void sample(@NotNull List<Sequence> sequences) {
        for (Sequence sequence : sequences) {
            int token = llama_sampler_sample(this.sampler, this.handle, sequence.outputIndex);

            sequence.accept(token, this.vocabulary, this.sequenceContextSize, System.nanoTime());
        }
    }

    private void checkOpen() {
        if (this.closed) {
            throw new IllegalStateException("The context is closed.");
        }

        this.model.checkOpen();
    }

    /**
     * Frees the context. Closing a closed context does nothing.
     */
    @Override
    public synchronized void close() {
        if (this.closed) {
            return;
        }

        this.closed = true;
        llama_sampler_free(this.sampler);
        llama_batch_ext_free(this.batch);
        llama_free(this.handle);
        this.arena.close();
        this.model.onContextClosed();
    }
}
