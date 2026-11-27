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

import java.util.List;
import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;

/**
 * The options of {@link LlamaModel#load(java.nio.file.Path, ModelOptions)}: how many layers go to GPUs and which GPUs hold them.
 * <p>
 * The defaults put all layers on every GPU device that llama.cpp found; without a GPU the model runs on the CPU. A model on the CPU
 * only uses {@code setGpuLayers(0)} and {@code setDevices(List.of())}.
 */
public final class ModelOptions {
    /**
     * The number of GPU layers that puts every layer of the model on the GPU devices.
     */
    public static final int ALL_LAYERS = -1;

    private int gpuLayers;
    private @Nullable List<LlamaDevice> devices;

    /**
     * Creates a {@link ModelOptions} instance with all layers on every GPU device.
     */
    public ModelOptions() {
        this.gpuLayers = ALL_LAYERS;
        this.devices   = null;
    }

    /**
     * @return the number of layers on the GPU devices, or {@link #ALL_LAYERS}
     */
    public int getGpuLayers() {
        return this.gpuLayers;
    }

    /**
     * Sets the number of layers on the GPU devices; the other layers run on the CPU.
     *
     * @param gpuLayers the number of layers, {@code 0} for none or {@link #ALL_LAYERS}
     * @return this instance for chaining
     * @throws IllegalArgumentException if the number is negative and not {@link #ALL_LAYERS}
     */
    public @NotNull ModelOptions setGpuLayers(int gpuLayers) {
        if (gpuLayers < ALL_LAYERS) {
            throw new IllegalArgumentException("Invalid number of GPU layers: " + gpuLayers);
        }

        this.gpuLayers = gpuLayers;
        return this;
    }

    /**
     * @return the GPU devices that hold the layers, or {@code null} for every GPU device
     */
    public @Nullable List<LlamaDevice> getDevices() {
        return this.devices;
    }

    /**
     * Sets the GPU devices that hold the layers.
     *
     * @param devices the GPU devices of {@link LlamaRuntime#getGpuDevices()}, an empty list for none or {@code null} for every GPU device
     * @return this instance for chaining
     * @throws IllegalArgumentException if a device is no GPU
     */
    public @NotNull ModelOptions setDevices(@Nullable List<LlamaDevice> devices) {
        if (devices != null) {
            for (LlamaDevice device : devices) {
                if (!device.getType().isGpu()) {
                    throw new IllegalArgumentException("The device " + device + " is no GPU.");
                }
            }
        }

        this.devices = devices == null ? null : List.copyOf(devices);
        return this;
    }
}
