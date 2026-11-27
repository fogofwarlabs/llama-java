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
 * Thrown when llama.cpp fails: the native libraries cannot be loaded, a model cannot be loaded, a context cannot be created or a batch
 * cannot be decoded.
 */
public final class LlamaException extends RuntimeException {
    private static final long serialVersionUID = 1L;

    /**
     * Creates a {@link LlamaException} instance with the given message.
     *
     * @param message the description of the failure
     */
    public LlamaException(@NotNull String message) {
        super(message);
    }

    /**
     * Creates a {@link LlamaException} instance with the given message and cause.
     *
     * @param message the description of the failure
     * @param cause   the exception that caused the failure
     */
    public LlamaException(@NotNull String message, @NotNull Throwable cause) {
        super(message, cause);
    }
}
