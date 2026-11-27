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

import static org.fogofwarlabs.llama.bindings.llama_h.C_LONG_LONG;
import static org.fogofwarlabs.llama.bindings.llama_h.ggml_backend_dev_memory;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import org.jetbrains.annotations.NotNull;

/**
 * A device of a ggml backend that llama.cpp can run a model on, for example the CPU or a Vulkan or Metal GPU.
 * <p>
 * The devices are those of the backends that loaded when the {@link LlamaRuntime} started; they live as long as the process.
 */
public final class LlamaDevice {
    private final @NotNull MemorySegment handle;

    private final @NotNull String name;
    private final @NotNull String description;
    private final @NotNull LlamaDeviceType type;
    private final @NotNull String backendName;

    /**
     * Creates a {@link LlamaDevice} instance for a {@code ggml_backend_dev_t}.
     *
     * @param handle      the native device
     * @param name        the device name of the backend, for example {@code Vulkan0}
     * @param description the description of the device, usually the product name
     * @param type        the device type
     * @param backendName the name of the backend, for example {@code Vulkan}
     */
    LlamaDevice(@NotNull MemorySegment handle,
                @NotNull String name, @NotNull String description, @NotNull LlamaDeviceType type, @NotNull String backendName) {
        this.handle = handle;

        this.name        = name;
        this.description = description;
        this.type        = type;
        this.backendName = backendName;
    }

    /**
     * @return the native {@code ggml_backend_dev_t}
     */
    @NotNull MemorySegment getHandle() {
        return this.handle;
    }

    /**
     * @return the device name of the backend, for example {@code Vulkan0} or {@code CPU}
     */
    public @NotNull String getName() {
        return this.name;
    }

    /**
     * @return the description of the device, usually the product name
     */
    public @NotNull String getDescription() {
        return this.description;
    }

    /**
     * @return the device type
     */
    public @NotNull LlamaDeviceType getType() {
        return this.type;
    }

    /**
     * @return the name of the backend of the device, for example {@code Vulkan}, {@code Metal} or {@code CPU}
     */
    public @NotNull String getBackendName() {
        return this.backendName;
    }

    /**
     * Reads the memory of the device that is free now.
     *
     * @return the free memory in bytes
     */
    public long getFreeMemory() {
        return this.readMemory(false);
    }

    /**
     * Reads the total memory of the device.
     *
     * @return the total memory in bytes
     */
    public long getTotalMemory() {
        return this.readMemory(true);
    }

    private long readMemory(boolean total) {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment free = arena.allocate(C_LONG_LONG);
            MemorySegment all = arena.allocate(C_LONG_LONG);

            ggml_backend_dev_memory(this.handle, free, all);

            return total ? all.get(C_LONG_LONG, 0) : free.get(C_LONG_LONG, 0);
        }
    }

    /**
     * Returns a string representation of the object.
     *
     * @return a string representation of the object
     */
    @Override
    public @NotNull String toString() {
        return this.name + " (" + this.description + ", " + this.type + ")";
    }
}
