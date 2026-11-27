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

import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.List;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

/**
 * Loads a model, reads its metadata and converts text to tokens and back; a missing model file fails and a closed model refuses access.
 */
final class LlamaModelTest {
    @Test
    void load_should_throw_when_the_model_file_is_missing() {
        Path missing = Path.of("missing-model.gguf");

        Assertions.assertThrows(LlamaException.class, () -> LlamaModel.load(missing, new ModelOptions()));
    }

    @Test
    void load_should_read_the_metadata_and_the_vocabulary() {
        Path path = LlamaTestResources.requireModel();
        ModelOptions options = new ModelOptions().setGpuLayers(0).setDevices(List.of());

        LlamaModel closedModel;

        try (LlamaModel model = LlamaModel.load(path, options)) {
            closedModel = model;

            LlamaVocabulary vocabulary = model.getVocabulary();
            String prompt = LlamaTestResources.createChatPrompt(LlamaTestResources.TRANSLATION_REQUEST);
            int[] tokens = vocabulary.tokenize(prompt, true, true);
            String chatTemplate = model.getChatTemplate();

            System.out.println(model.getDescription() + ", " + model.getParameterCount() + " parameters, " + model.getSize() + " bytes, " +
                               "trained context " + model.getTrainingContextSize() + ", " + vocabulary.getTokenCount() + " tokens");

            Assertions.assertFalse(model.getDescription().isEmpty());
            Assertions.assertTrue(model.getSize() > 0);
            Assertions.assertTrue(model.getTrainingContextSize() > 0);
            Assertions.assertNotNull(chatTemplate);
            Assertions.assertTrue(chatTemplate.contains("<|im_start|>"));
            Assertions.assertTrue(vocabulary.isEndOfGeneration(vocabulary.tokenize("<|im_end|>", false, true)[0]));
            Assertions.assertTrue(tokens.length > 0 && tokens.length < prompt.length());
            Assertions.assertEquals(prompt, vocabulary.detokenize(tokens, false, true));
            Assertions.assertEquals("<|im_start|>", new String(vocabulary.getTokenPiece(tokens[0], true), StandardCharsets.UTF_8));
        }

        Assertions.assertThrows(IllegalStateException.class, () -> closedModel.getVocabulary().getTokenCount());
    }
}
