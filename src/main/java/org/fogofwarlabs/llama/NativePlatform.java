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
import java.util.Locale;
import org.jetbrains.annotations.NotNull;

/**
 * The platforms with native jars, each with the libraries that are loaded before the llama.cpp API is used, in dependency order.
 * <p>
 * The CPU variants and the GPU backend of Windows and Linux are not listed: llama.cpp loads them at run time from the same folder.
 */
enum NativePlatform {
    WINDOWS_X86_64("windows-x86_64", List.of("libomp.dll", "ggml-base.dll", "ggml.dll", "llama.dll")),
    LINUX_X86_64("linux-x86_64", List.of("libggml-base.so.1", "libggml.so.1", "libllama.so.1")),
    MACOSX_ARM64("macosx-arm64", List.of("libggml-base.1.dylib", "libggml-cpu.1.dylib", "libggml-blas.1.dylib", "libggml-metal.1.dylib",
                                         "libggml-rpc.1.dylib", "libggml.1.dylib", "libllama.1.dylib"));

    /**
     * Finds the platform of the running Java virtual machine.
     *
     * @return the platform of this machine
     * @throws LlamaException if no native jar exists for this operating system and architecture
     */
    static @NotNull NativePlatform detect() {
        String operatingSystem = System.getProperty("os.name").toLowerCase(Locale.ROOT);
        String architecture = System.getProperty("os.arch").toLowerCase(Locale.ROOT);
        boolean x86 = architecture.equals("amd64") || architecture.equals("x86_64");
        boolean arm = architecture.equals("aarch64") || architecture.equals("arm64");

        if (operatingSystem.startsWith("windows") && x86) {
            return WINDOWS_X86_64;
        }

        if (operatingSystem.startsWith("linux") && x86) {
            return LINUX_X86_64;
        }

        if (operatingSystem.startsWith("mac") && arm) {
            return MACOSX_ARM64;
        }

        throw new LlamaException("No native libraries exist for " + operatingSystem + " on " + architecture + ".");
    }

    private final @NotNull String classifier;
    private final @NotNull List<String> preloadedLibraries;

    NativePlatform(@NotNull String classifier, @NotNull List<String> preloadedLibraries) {
        this.classifier         = classifier;
        this.preloadedLibraries = preloadedLibraries;
    }

    /**
     * @return the classifier of the native jar, also the folder of its libraries below {@code org/fogofwarlabs/llama/}
     */
    @NotNull String getClassifier() {
        return this.classifier;
    }

    /**
     * @return the file names of the libraries to load before the API is used, in dependency order
     */
    @NotNull List<String> getPreloadedLibraries() {
        return this.preloadedLibraries;
    }
}
