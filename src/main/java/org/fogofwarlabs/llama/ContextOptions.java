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

import org.jetbrains.annotations.NotNull;

/**
 * The options of {@link LlamaModel#createContext(ContextOptions)}: the sizes of the context memory and of the batches, the number of
 * parallel sequences and the CPU threads.
 */
public final class ContextOptions {
    /**
     * The default context size in tokens, shared by all sequences.
     */
    public static final int DEFAULT_CONTEXT_SIZE = 2048;

    /**
     * The default number of tokens that one decoding step takes.
     */
    public static final int DEFAULT_BATCH_SIZE = 512;

    /**
     * The default number of tokens that the backend computes at once.
     */
    public static final int DEFAULT_MICRO_BATCH_SIZE = 512;

    /**
     * The default number of sequences that are generated in parallel.
     */
    public static final int DEFAULT_SEQUENCE_COUNT = 1;

    /**
     * The default number of CPU threads, for generation and for prompts.
     */
    public static final int DEFAULT_THREAD_COUNT = 4;

    private static int requirePositive(int value, @NotNull String name) {
        if (value <= 0) {
            throw new IllegalArgumentException("Invalid " + name + ": " + value);
        }

        return value;
    }

    private int contextSize;
    private int batchSize;
    private int microBatchSize;
    private int sequenceCount;
    private boolean unifiedMemory;

    private int threadCount;
    private int batchThreadCount;

    /**
     * Creates a {@link ContextOptions} instance with the default sizes, one sequence and {@value #DEFAULT_THREAD_COUNT} threads.
     */
    public ContextOptions() {
        this.contextSize    = DEFAULT_CONTEXT_SIZE;
        this.batchSize      = DEFAULT_BATCH_SIZE;
        this.microBatchSize = DEFAULT_MICRO_BATCH_SIZE;
        this.sequenceCount  = DEFAULT_SEQUENCE_COUNT;
        this.unifiedMemory  = false;

        this.threadCount      = DEFAULT_THREAD_COUNT;
        this.batchThreadCount = DEFAULT_THREAD_COUNT;
    }

    /**
     * @return the context size in tokens, shared by all sequences
     */
    public int getContextSize() {
        return this.contextSize;
    }

    /**
     * Sets the context size in tokens, shared by all sequences: prompt and generated tokens of one request must fit into it.
     *
     * @param contextSize the context size
     * @return this instance for chaining
     * @throws IllegalArgumentException if the size is not positive
     */
    public @NotNull ContextOptions setContextSize(int contextSize) {
        this.contextSize = requirePositive(contextSize, "context size");
        return this;
    }

    /**
     * @return the number of tokens that one decoding step takes
     */
    public int getBatchSize() {
        return this.batchSize;
    }

    /**
     * Sets the number of tokens that one decoding step takes; longer prompts take several steps.
     *
     * @param batchSize the batch size
     * @return this instance for chaining
     * @throws IllegalArgumentException if the size is not positive
     */
    public @NotNull ContextOptions setBatchSize(int batchSize) {
        this.batchSize = requirePositive(batchSize, "batch size");
        return this;
    }

    /**
     * @return the number of tokens that the backend computes at once
     */
    public int getMicroBatchSize() {
        return this.microBatchSize;
    }

    /**
     * Sets the number of tokens that the backend computes at once; a smaller size needs less compute memory.
     *
     * @param microBatchSize the micro-batch size
     * @return this instance for chaining
     * @throws IllegalArgumentException if the size is not positive
     */
    public @NotNull ContextOptions setMicroBatchSize(int microBatchSize) {
        this.microBatchSize = requirePositive(microBatchSize, "micro-batch size");
        return this;
    }

    /**
     * @return the number of sequences that can be generated in parallel
     */
    public int getSequenceCount() {
        return this.sequenceCount;
    }

    /**
     * Sets the number of sequences that can be generated in parallel by
     * {@link LlamaContext#generate(java.util.List, int)}.
     *
     * @param sequenceCount the number of sequences
     * @return this instance for chaining
     * @throws IllegalArgumentException if the number is not positive
     */
    public @NotNull ContextOptions setSequenceCount(int sequenceCount) {
        this.sequenceCount = requirePositive(sequenceCount, "sequence count");
        return this;
    }

    /**
     * @return {@code true} if the sequences share one memory of the whole context size, {@code false} if each sequence has an equal part
     */
    public boolean isUnifiedMemory() {
        return this.unifiedMemory;
    }

    /**
     * Sets whether the sequences share one memory of the whole context size, which suits sequences of different lengths, or each sequence
     * has an equal part of it.
     *
     * @param unifiedMemory {@code true} for one shared memory
     * @return this instance for chaining
     */
    public @NotNull ContextOptions setUnifiedMemory(boolean unifiedMemory) {
        this.unifiedMemory = unifiedMemory;
        return this;
    }

    /**
     * @return the number of CPU threads for generation
     */
    public int getThreadCount() {
        return this.threadCount;
    }

    /**
     * Sets the number of CPU threads for generation, one token per sequence and step.
     *
     * @param threadCount the number of threads
     * @return this instance for chaining
     * @throws IllegalArgumentException if the number is not positive
     */
    public @NotNull ContextOptions setThreadCount(int threadCount) {
        this.threadCount = requirePositive(threadCount, "thread count");
        return this;
    }

    /**
     * @return the number of CPU threads for prompts
     */
    public int getBatchThreadCount() {
        return this.batchThreadCount;
    }

    /**
     * Sets the number of CPU threads for prompts and other steps of several tokens.
     *
     * @param batchThreadCount the number of threads
     * @return this instance for chaining
     * @throws IllegalArgumentException if the number is not positive
     */
    public @NotNull ContextOptions setBatchThreadCount(int batchThreadCount) {
        this.batchThreadCount = requirePositive(batchThreadCount, "batch thread count");
        return this;
    }
}
