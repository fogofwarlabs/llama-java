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

import static org.fogofwarlabs.llama.bindings.llama_h.GGML_BACKEND_DEVICE_TYPE_ACCEL;
import static org.fogofwarlabs.llama.bindings.llama_h.GGML_BACKEND_DEVICE_TYPE_CPU;
import static org.fogofwarlabs.llama.bindings.llama_h.GGML_BACKEND_DEVICE_TYPE_GPU;
import static org.fogofwarlabs.llama.bindings.llama_h.GGML_BACKEND_DEVICE_TYPE_IGPU;
import static org.fogofwarlabs.llama.bindings.llama_h.GGML_BACKEND_DEVICE_TYPE_META;

import org.jetbrains.annotations.NotNull;

/**
 * The kinds of devices that the ggml backends offer.
 */
public enum LlamaDeviceType {
    /**
     * The processor, using system memory.
     */
    CPU,

    /**
     * A GPU with dedicated memory.
     */
    GPU,

    /**
     * An integrated GPU, using system memory.
     */
    INTEGRATED_GPU,

    /**
     * An accelerator that works together with the CPU backend, for example BLAS.
     */
    ACCELERATOR,

    /**
     * A device that wraps several other devices.
     */
    META;

    /**
     * Finds the type of a {@code ggml_backend_dev_type} value.
     *
     * @param value the native device type
     * @return the device type
     * @throws LlamaException if the value is no known device type
     */
    static @NotNull LlamaDeviceType fromNative(int value) {
        if (value == GGML_BACKEND_DEVICE_TYPE_CPU()) {
            return CPU;
        }

        if (value == GGML_BACKEND_DEVICE_TYPE_GPU()) {
            return GPU;
        }

        if (value == GGML_BACKEND_DEVICE_TYPE_IGPU()) {
            return INTEGRATED_GPU;
        }

        if (value == GGML_BACKEND_DEVICE_TYPE_ACCEL()) {
            return ACCELERATOR;
        }

        if (value == GGML_BACKEND_DEVICE_TYPE_META()) {
            return META;
        }

        throw new LlamaException("Unknown device type " + value + ".");
    }

    /**
     * @return {@code true} if the device is a GPU, dedicated or integrated, {@code false} otherwise
     */
    public boolean isGpu() {
        return this == GPU || this == INTEGRATED_GPU;
    }
}
