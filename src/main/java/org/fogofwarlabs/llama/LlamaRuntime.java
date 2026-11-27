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

import static org.fogofwarlabs.llama.bindings.llama_h.ggml_backend_dev_backend_reg;
import static org.fogofwarlabs.llama.bindings.llama_h.ggml_backend_dev_count;
import static org.fogofwarlabs.llama.bindings.llama_h.ggml_backend_dev_description;
import static org.fogofwarlabs.llama.bindings.llama_h.ggml_backend_dev_get;
import static org.fogofwarlabs.llama.bindings.llama_h.ggml_backend_dev_name;
import static org.fogofwarlabs.llama.bindings.llama_h.ggml_backend_dev_type;
import static org.fogofwarlabs.llama.bindings.llama_h.ggml_backend_load_all_from_path;
import static org.fogofwarlabs.llama.bindings.llama_h.ggml_backend_reg_name;
import static org.fogofwarlabs.llama.bindings.llama_h.llama_backend_init;
import static org.fogofwarlabs.llama.bindings.llama_h.llama_print_system_info;

import java.io.IOException;
import java.io.InputStream;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Properties;
import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;

/**
 * The loaded llama.cpp libraries of this process and the devices of their backends.
 * <p>
 * {@link #getInstance()} loads the native libraries once: from the folder named by the system property
 * {@code org.fogofwarlabs.llama.librarypath}, or from the native jar of this platform on the class path, extracted below the folder named
 * by {@code org.fogofwarlabs.llama.cachedir} (by default {@code .llama-java/cache} in the user's home folder). It then loads the ggml
 * backends of that folder: the CPU variant that fits the processor and the GPU backend of the platform. A GPU backend whose driver is
 * missing is left out, and the CPU device remains.
 * <p>
 * The native log goes to the {@link System.Logger} {@code org.fogofwarlabs.llama}.
 */
public final class LlamaRuntime {
    private static final String BUILD_RESOURCE = "/org/fogofwarlabs/llama/llama-java.properties";
    private static final String BUILD_PROPERTY = "llama.build";

    private static @Nullable LlamaRuntime instance;

    /**
     * Loads the native libraries on the first call and returns the runtime.
     *
     * @return the runtime of this process
     * @throws LlamaException if the native libraries cannot be found or loaded
     */
    public static synchronized @NotNull LlamaRuntime getInstance() {
        if (instance == null) {
            instance = new LlamaRuntime();
        }

        return instance;
    }

    private static @NotNull String readBuild() {
        Properties properties = new Properties();

        try (InputStream input = LlamaRuntime.class.getResourceAsStream(BUILD_RESOURCE)) {
            if (input == null) {
                throw new LlamaException("The resource " + BUILD_RESOURCE + " is missing.");
            }

            properties.load(input);
        } catch (IOException exception) {
            throw new LlamaException("Cannot read the resource " + BUILD_RESOURCE + ": " + exception.getMessage(), exception);
        }

        return properties.getProperty(BUILD_PROPERTY, "");
    }

    private static @NotNull List<LlamaDevice> readDevices() {
        long count = ggml_backend_dev_count();
        List<LlamaDevice> devices = new ArrayList<>();

        for (long index = 0; index < count; index++) {
            MemorySegment device = ggml_backend_dev_get(index);
            String name = ggml_backend_dev_name(device).getString(0);
            String description = ggml_backend_dev_description(device).getString(0);
            LlamaDeviceType type = LlamaDeviceType.fromNative(ggml_backend_dev_type(device));
            String backendName = ggml_backend_reg_name(ggml_backend_dev_backend_reg(device)).getString(0);

            devices.add(new LlamaDevice(device, name, description, type, backendName));
        }

        return List.copyOf(devices);
    }

    private final @NotNull String build;
    private final @NotNull NativePlatform platform;
    private final @NotNull Path libraryDirectory;

    private final @NotNull NativeLogForwarder logForwarder;
    private final @NotNull List<LlamaDevice> devices;

    private LlamaRuntime() {
        this.build            = readBuild();
        this.platform         = NativePlatform.detect();
        this.libraryDirectory = NativeLibraryUtils.loadLibraries(this.platform, this.build);

        // The log callback comes first: loading the backends already writes to the log.
        this.logForwarder = new NativeLogForwarder();
        this.logForwarder.install();

        llama_backend_init();

        try (Arena arena = Arena.ofConfined()) {
            ggml_backend_load_all_from_path(arena.allocateFrom(this.libraryDirectory.toString()));
        }

        this.devices = readDevices();
    }

    /**
     * @return the llama.cpp release build of the native libraries, for example {@code b12811}
     */
    public @NotNull String getBuild() {
        return this.build;
    }

    /**
     * @return the platform classifier of the native libraries, for example {@code windows-x86_64}
     */
    public @NotNull String getPlatform() {
        return this.platform.getClassifier();
    }

    /**
     * @return the folder that holds the loaded native libraries
     */
    public @NotNull Path getLibraryDirectory() {
        return this.libraryDirectory;
    }

    /**
     * @return the devices of the loaded backends, in the order of llama.cpp
     */
    public @NotNull List<LlamaDevice> getDevices() {
        return this.devices;
    }

    /**
     * Finds the devices that can hold model layers: dedicated and integrated GPUs.
     *
     * @return the GPU devices, in the order of llama.cpp
     */
    public @NotNull List<LlamaDevice> getGpuDevices() {
        return this.devices.stream().filter(device -> device.getType().isGpu()).toList();
    }

    /**
     * Reads the system information of llama.cpp: the instruction set features of the loaded CPU backend and the backends.
     *
     * @return the system information line
     */
    public @NotNull String getSystemInfo() {
        return llama_print_system_info().getString(0).strip();
    }
}
