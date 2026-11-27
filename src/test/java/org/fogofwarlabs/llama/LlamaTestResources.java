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

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import org.jetbrains.annotations.NotNull;
import org.junit.jupiter.api.Assumptions;

/**
 * The model of the model tests and its prompts. The model is a GGUF file of Index-Translate-2B, a model in the Qwen chat format, named by
 * the system property {@value #MODEL_PROPERTY}; the tests that need it are skipped without it.
 */
final class LlamaTestResources {
    /**
     * The system property that names the GGUF file of the model tests.
     */
    static final String MODEL_PROPERTY = "llama.test.model";

    /**
     * The translation request of the smoke test: Korean chat to English.
     */
    static final String TRANSLATION_REQUEST = "请将以下文本翻译为英语，直接输出翻译结果，不要进行任何解释：\n\n미네랄 좀 보내줄 수 있어?";

    /**
     * A second translation request for the parallel sequences: Korean chat to English.
     */
    static final String SECOND_TRANSLATION_REQUEST = "请将以下文本翻译为英语，直接输出翻译结果，不要进行任何解释：\n\n질럿 두 기만 더 뽑아 줘.";

    /**
     * The word that the English translation of {@link #TRANSLATION_REQUEST} holds.
     */
    static final String TRANSLATION_WORD = "minerals";

    /**
     * The maximum number of generated tokens of a translation.
     */
    static final int TRANSLATION_MAXIMUM_TOKENS = 64;

    private static final String CHAT_START = "<|im_start|>user\n";
    private static final String CHAT_END = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";

    /**
     * Finds the model file, skipping the calling test when none is given.
     *
     * @return the GGUF file
     */
    static @NotNull Path requireModel() {
        String model = System.getProperty(MODEL_PROPERTY, "");

        Assumptions.assumeTrue(!model.isEmpty() && Files.isRegularFile(Path.of(model)), "no model file in " + MODEL_PROPERTY);
        return Path.of(model);
    }

    /**
     * Writes a user message in the Qwen chat format with thinking turned off.
     *
     * @param message the user message
     * @return the prompt
     */
    static @NotNull String createChatPrompt(@NotNull String message) {
        return CHAT_START + message + CHAT_END;
    }

    /**
     * Finds the first GPU device, skipping the calling test when there is none.
     *
     * @return the GPU device
     */
    static @NotNull LlamaDevice requireGpuDevice() {
        List<LlamaDevice> devices = LlamaRuntime.getInstance().getGpuDevices();

        Assumptions.assumeFalse(devices.isEmpty(), "no GPU device");
        return devices.getFirst();
    }

    private LlamaTestResources() {
    }
}
