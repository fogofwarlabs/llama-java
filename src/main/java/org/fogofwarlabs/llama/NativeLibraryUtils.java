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

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.net.URL;
import java.net.URLConnection;
import java.nio.charset.StandardCharsets;
import java.nio.file.AtomicMoveNotSupportedException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.ArrayList;
import java.util.List;
import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;

/**
 * Finds the native libraries of this platform and loads the ones that the API links to.
 * <p>
 * The libraries come from the folder named by the system property {@value #LIBRARY_PATH_PROPERTY} when it is set, otherwise from the
 * native jar of this platform on the class path: its libraries are extracted to
 * {@code <cache folder>/<llama.cpp build>/<platform>/}, where the cache folder is named by the system property
 * {@value #CACHE_DIRECTORY_PROPERTY} or is {@code .llama-java/cache} in the user's home folder. A library that already exists there with
 * the same size is kept.
 */
final class NativeLibraryUtils {
    /**
     * The system property that names a folder holding the native libraries, for example the flat native folder of an application.
     */
    static final String LIBRARY_PATH_PROPERTY = "org.fogofwarlabs.llama.librarypath";

    /**
     * The system property that names the folder below which the libraries of the native jar are extracted.
     */
    static final String CACHE_DIRECTORY_PROPERTY = "org.fogofwarlabs.llama.cachedir";

    private static final String RESOURCE_FOLDER = "org/fogofwarlabs/llama/";
    private static final String LIBRARY_LIST = "libraries.list";
    private static final String DEFAULT_CACHE_FOLDER = ".llama-java";
    private static final String DEFAULT_CACHE_SUBFOLDER = "cache";
    private static final String TEMPORARY_SUFFIX = ".tmp";

    /**
     * Finds the native libraries, extracting them from the class path if needed, and loads the libraries of the API in dependency order.
     *
     * @param platform the platform of this machine
     * @param build    the llama.cpp build of the libraries, which names the folder of the extracted libraries
     * @return the folder that holds the libraries, which llama.cpp searches for its backends
     * @throws LlamaException if the libraries cannot be found, extracted or loaded
     */
    static @NotNull Path loadLibraries(@NotNull NativePlatform platform, @NotNull String build) {
        String libraryPath = System.getProperty(LIBRARY_PATH_PROPERTY, "");
        Path directory = libraryPath.isEmpty() ? extractLibraries(platform, build) : Path.of(libraryPath).toAbsolutePath();

        for (String name : platform.getPreloadedLibraries()) {
            Path library = directory.resolve(name);

            if (!Files.isRegularFile(library)) {
                throw new LlamaException("The native library " + name + " is missing in " + directory + ".");
            }

            try {
                System.load(library.toString());
            } catch (UnsatisfiedLinkError error) {
                throw new LlamaException("Cannot load the native library " + library + ": " + error.getMessage(), error);
            }
        }

        return directory;
    }

    private static @NotNull Path extractLibraries(@NotNull NativePlatform platform, @NotNull String build) {
        String resourceFolder = RESOURCE_FOLDER + platform.getClassifier() + "/";
        List<String> names = readLibraryList(resourceFolder + LIBRARY_LIST);

        if (names == null) {
            throw new LlamaException("The native jar org.fogofwarlabs:llama-java:<version>:" + platform.getClassifier() + " is not on the class path, " +
                                     "and the system property " + LIBRARY_PATH_PROPERTY + " names no folder of native libraries.");
        }

        String cacheDirectory = System.getProperty(CACHE_DIRECTORY_PROPERTY, "");
        Path cacheRoot = cacheDirectory.isEmpty()
                         ? Path.of(System.getProperty("user.home"), DEFAULT_CACHE_FOLDER, DEFAULT_CACHE_SUBFOLDER)
                         : Path.of(cacheDirectory);
        Path directory = cacheRoot.resolve(build).resolve(platform.getClassifier()).toAbsolutePath();

        try {
            Files.createDirectories(directory);

            for (String name : names) {
                extractLibrary(resourceFolder + name, directory.resolve(name));
            }
        } catch (IOException exception) {
            throw new LlamaException("Cannot extract the native libraries to " + directory + ": " + exception.getMessage(), exception);
        }

        return directory;
    }

    private static @Nullable List<String> readLibraryList(@NotNull String resource) {
        InputStream stream = NativeLibraryUtils.class.getClassLoader().getResourceAsStream(resource);

        if (stream == null) {
            return null;
        }

        List<String> names = new ArrayList<>();

        try (BufferedReader reader = new BufferedReader(new InputStreamReader(stream, StandardCharsets.UTF_8))) {
            String line;

            while ((line = reader.readLine()) != null) {
                String name = line.strip();

                if (!name.isEmpty()) {
                    names.add(name);
                }
            }
        } catch (IOException exception) {
            throw new LlamaException("Cannot read the resource " + resource + ": " + exception.getMessage(), exception);
        }

        return names;
    }

    private static void extractLibrary(@NotNull String resource, @NotNull Path target) throws IOException {
        URL url = NativeLibraryUtils.class.getClassLoader().getResource(resource);

        if (url == null) {
            throw new LlamaException("The resource " + resource + " of the native jar is missing.");
        }

        URLConnection connection = url.openConnection();
        long size = connection.getContentLengthLong();

        if (size >= 0 && Files.isRegularFile(target) && Files.size(target) == size) {
            return;
        }

        // Another process may extract or use the same library: the copy goes to a file of its own and replaces the target in one step.
        Path temporary = Files.createTempFile(target.getParent(), target.getFileName().toString(), TEMPORARY_SUFFIX);

        try {
            try (InputStream input = connection.getInputStream()) {
                Files.copy(input, temporary, StandardCopyOption.REPLACE_EXISTING);
            }

            try {
                Files.move(temporary, target, StandardCopyOption.REPLACE_EXISTING, StandardCopyOption.ATOMIC_MOVE);
            } catch (AtomicMoveNotSupportedException exception) {
                Files.move(temporary, target, StandardCopyOption.REPLACE_EXISTING);
            } catch (IOException exception) {
                // A library that another process has loaded cannot be replaced on Windows; it is the same file when the sizes match.
                if (!Files.isRegularFile(target) || Files.size(target) != Files.size(temporary)) {
                    throw exception;
                }
            }
        } finally {
            Files.deleteIfExists(temporary);
        }
    }

    private NativeLibraryUtils() {
    }
}
