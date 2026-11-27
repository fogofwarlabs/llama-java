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

import java.nio.file.Path;
import java.util.List;
import java.util.Locale;
import org.jetbrains.annotations.NotNull;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

/**
 * Translates Korean game chat to English with greedy generation on the CPU and on the first GPU, one prompt and two parallel prompts.
 */
final class LlamaContextTest {
    private static final int CPU_THREAD_COUNT = 4;
    private static final int PARALLEL_SEQUENCE_COUNT = 2;
    private static final double NANOSECONDS_PER_MILLISECOND = 1_000_000.0;

    @Test
    void generate_should_translate_on_the_cpu() {
        ModelOptions options = new ModelOptions().setGpuLayers(0).setDevices(List.of());

        translate("cpu", options);
    }

    @Test
    void generate_should_translate_on_the_gpu() {
        Path path = LlamaTestResources.requireModel();
        LlamaDevice device = LlamaTestResources.requireGpuDevice();
        ModelOptions options = new ModelOptions().setGpuLayers(ModelOptions.ALL_LAYERS).setDevices(List.of(device));

        System.out.println("GPU: " + device + " of " + path.getFileName());
        translate(device.getBackendName().toLowerCase(Locale.ROOT), options);
    }

    private static void translate(@NotNull String label, @NotNull ModelOptions modelOptions) {
        Path path = LlamaTestResources.requireModel();
        ContextOptions contextOptions = new ContextOptions().setSequenceCount(PARALLEL_SEQUENCE_COUNT)
                                                            .setUnifiedMemory(true)
                                                            .setThreadCount(CPU_THREAD_COUNT)
                                                            .setBatchThreadCount(CPU_THREAD_COUNT);
        String prompt = LlamaTestResources.createChatPrompt(LlamaTestResources.TRANSLATION_REQUEST);
        String secondPrompt = LlamaTestResources.createChatPrompt(LlamaTestResources.SECOND_TRANSLATION_REQUEST);

        long loadStart = System.nanoTime();

        try (LlamaModel model = LlamaModel.load(path, modelOptions);
             LlamaContext context = model.createContext(contextOptions)) {
            long warmUpStart = System.nanoTime();

            context.warmUp();

            long warmUpEnd = System.nanoTime();

            System.out.printf("%s: load %.0f ms, warm-up %.0f ms%n",
                              label, (warmUpStart - loadStart) / NANOSECONDS_PER_MILLISECOND, (warmUpEnd - warmUpStart) / NANOSECONDS_PER_MILLISECOND);

            Generation first = context.generate(prompt, LlamaTestResources.TRANSLATION_MAXIMUM_TOKENS);
            Generation second = context.generate(prompt, LlamaTestResources.TRANSLATION_MAXIMUM_TOKENS);
            List<Generation> parallel = context.generate(List.of(prompt, secondPrompt), LlamaTestResources.TRANSLATION_MAXIMUM_TOKENS);

            print(label + " first", first);
            print(label + " second", second);
            print(label + " parallel 1", parallel.get(0));
            print(label + " parallel 2", parallel.get(1));

            Assertions.assertTrue(first.getText().toLowerCase(Locale.ROOT).contains(LlamaTestResources.TRANSLATION_WORD), first.getText());
            Assertions.assertEquals(StopReason.END_OF_GENERATION, first.getStopReason());
            Assertions.assertEquals(first.getText(), second.getText());
            Assertions.assertEquals(first.getText(), parallel.get(0).getText());
            Assertions.assertFalse(parallel.get(1).getText().isBlank());
            Assertions.assertEquals(StopReason.END_OF_GENERATION, parallel.get(1).getStopReason());
        }
    }

    private static void print(@NotNull String label, @NotNull Generation generation) {
        System.out.printf("%s: \"%s\" (%s), prompt %d tokens in %.1f ms, generated %d tokens in %.1f ms%n",
                          label, generation.getText(), generation.getStopReason(),
                          generation.getPromptTokenCount(), generation.getPromptDuration().toNanos() / NANOSECONDS_PER_MILLISECOND,
                          generation.getGeneratedTokenCount(), generation.getGenerationDuration().toNanos() / NANOSECONDS_PER_MILLISECOND);
    }
}
