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
import static org.fogofwarlabs.llama.bindings.llama_h.C_POINTER;
import static org.fogofwarlabs.llama.bindings.llama_h.llama_model_chat_template;
import static org.fogofwarlabs.llama.bindings.llama_h.llama_model_default_params;
import static org.fogofwarlabs.llama.bindings.llama_h.llama_model_desc;
import static org.fogofwarlabs.llama.bindings.llama_h.llama_model_free;
import static org.fogofwarlabs.llama.bindings.llama_h.llama_model_get_vocab;
import static org.fogofwarlabs.llama.bindings.llama_h.llama_model_load_from_file;
import static org.fogofwarlabs.llama.bindings.llama_h.llama_model_n_ctx_train;
import static org.fogofwarlabs.llama.bindings.llama_h.llama_model_n_params;
import static org.fogofwarlabs.llama.bindings.llama_h.llama_model_size;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import org.fogofwarlabs.llama.bindings.llama_model_params;
import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;

/**
 * A model loaded from a GGUF file. It owns the weights in system and device memory until {@link #close()}; its contexts must be closed
 * before it.
 */
public final class LlamaModel implements AutoCloseable {
    private static final int DESCRIPTION_CAPACITY = 256;

    /**
     * Loads a model from a GGUF file.
     *
     * @param path    the GGUF file
     * @param options the layers and devices of the model
     * @return the loaded model
     * @throws LlamaException if the native libraries cannot be loaded or the model cannot be loaded
     */
    public static @NotNull LlamaModel load(@NotNull Path path, @NotNull ModelOptions options) {
        LlamaRuntime.getInstance();

        if (!Files.isRegularFile(path)) {
            throw new LlamaException("The model file " + path + " does not exist.");
        }

        MemorySegment handle;

        try (Arena arena = Arena.ofConfined()) {
            MemorySegment parameters = llama_model_default_params(arena);
            List<LlamaDevice> devices = options.getDevices();

            llama_model_params.n_gpu_layers(parameters, options.getGpuLayers());

            if (devices != null) {
                // A list that ends with NULL; an empty list keeps the model on the CPU.
                MemorySegment deviceList = arena.allocate(C_POINTER, devices.size() + 1L);

                for (int index = 0; index < devices.size(); index++) {
                    deviceList.setAtIndex(C_POINTER, index, devices.get(index).getHandle());
                }

                llama_model_params.devices(parameters, deviceList);
            }

            handle = llama_model_load_from_file(arena.allocateFrom(path.toAbsolutePath().toString()), parameters);
        }

        if (handle.equals(MemorySegment.NULL)) {
            throw new LlamaException("Cannot load the model " + path + "; the log org.fogofwarlabs.llama names the cause.");
        }

        return new LlamaModel(path, handle);
    }

    private final @NotNull Path path;
    private final @NotNull MemorySegment handle;
    private final @NotNull LlamaVocabulary vocabulary;

    private int openContexts;
    private boolean closed;

    private LlamaModel(@NotNull Path path, @NotNull MemorySegment handle) {
        this.path       = path;
        this.handle     = handle;
        this.vocabulary = new LlamaVocabulary(this, llama_model_get_vocab(handle));

        this.openContexts = 0;
        this.closed       = false;
    }

    /**
     * @return the native {@code llama_model}
     */
    @NotNull MemorySegment getHandle() {
        return this.handle;
    }

    /**
     * @return the GGUF file of the model
     */
    public @NotNull Path getPath() {
        return this.path;
    }

    /**
     * @return the vocabulary of the model
     */
    public @NotNull LlamaVocabulary getVocabulary() {
        return this.vocabulary;
    }

    /**
     * Reads the short description of the model: architecture, size and quantization.
     *
     * @return the description, for example {@code qwen3 1.7B Q4_K - Medium}
     */
    public @NotNull String getDescription() {
        this.checkOpen();

        try (Arena arena = Arena.ofConfined()) {
            MemorySegment buffer = arena.allocate(DESCRIPTION_CAPACITY);
            int length = llama_model_desc(this.handle, buffer, DESCRIPTION_CAPACITY);

            return new String(buffer.asSlice(0, Math.clamp(length, 0, DESCRIPTION_CAPACITY - 1)).toArray(C_CHAR), StandardCharsets.UTF_8);
        }
    }

    /**
     * Reads the default chat template of the model from its metadata ({@code tokenizer.chat_template}).
     *
     * @return the Jinja template, or {@code null} if the model has none
     */
    public @Nullable String getChatTemplate() {
        this.checkOpen();

        MemorySegment template = llama_model_chat_template(this.handle, MemorySegment.NULL);

        return template.equals(MemorySegment.NULL) ? null : template.getString(0);
    }

    /**
     * @return the size of the weights in bytes
     */
    public long getSize() {
        this.checkOpen();
        return llama_model_size(this.handle);
    }

    /**
     * @return the number of parameters of the model
     */
    public long getParameterCount() {
        this.checkOpen();
        return llama_model_n_params(this.handle);
    }

    /**
     * @return the context size that the model was trained with
     */
    public int getTrainingContextSize() {
        this.checkOpen();
        return llama_model_n_ctx_train(this.handle);
    }

    /**
     * Creates a context on this model.
     *
     * @param options the sizes and threads of the context
     * @return the created {@link LlamaContext} instance
     * @throws LlamaException if the context cannot be created
     */
    public synchronized @NotNull LlamaContext createContext(@NotNull ContextOptions options) {
        this.checkOpen();

        LlamaContext context = new LlamaContext(this, options);

        this.openContexts++;
        return context;
    }

    /**
     * Called when a context of this model is closed.
     */
    synchronized void onContextClosed() {
        this.openContexts--;
    }

    /**
     * Checks that the model is open.
     *
     * @throws IllegalStateException if the model is closed
     */
    synchronized void checkOpen() {
        if (this.closed) {
            throw new IllegalStateException("The model " + this.path + " is closed.");
        }
    }

    /**
     * Frees the model. Closing a closed model does nothing.
     *
     * @throws IllegalStateException if a context of the model is open
     */
    @Override
    public synchronized void close() {
        if (this.closed) {
            return;
        }

        if (this.openContexts > 0) {
            throw new IllegalStateException("The model " + this.path + " has " + this.openContexts + " open contexts.");
        }

        this.closed = true;
        llama_model_free(this.handle);
    }
}
