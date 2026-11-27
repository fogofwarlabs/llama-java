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

import java.time.Duration;
import org.jetbrains.annotations.NotNull;

/**
 * The result of generating one sequence: the generated text and tokens, why the generation stopped and how long it took.
 * <p>
 * The prompt time is that of all prompts of a request, which are decoded together; the generation time runs from the end of the prompts to
 * the stop of this sequence. The end-of-generation token that stops a sequence is not part of its tokens and text.
 */
public final class Generation {
    private final @NotNull String text;
    private final int @NotNull [] tokens;
    private final @NotNull StopReason stopReason;

    private final int promptTokenCount;
    private final @NotNull Duration promptDuration;
    private final @NotNull Duration generationDuration;

    /**
     * Creates a {@link Generation} instance.
     *
     * @param text               the generated text
     * @param tokens             the generated tokens
     * @param stopReason         why the generation stopped
     * @param promptTokenCount   the number of tokens of the prompt
     * @param promptDuration     the time of decoding the prompts of the request
     * @param generationDuration the time from the end of the prompts to the stop of this sequence
     */
    Generation(@NotNull String text, int @NotNull [] tokens, @NotNull StopReason stopReason,
               int promptTokenCount, @NotNull Duration promptDuration, @NotNull Duration generationDuration) {
        this.text       = text;
        this.tokens     = tokens;
        this.stopReason = stopReason;

        this.promptTokenCount   = promptTokenCount;
        this.promptDuration     = promptDuration;
        this.generationDuration = generationDuration;
    }

    /**
     * @return the generated text
     */
    public @NotNull String getText() {
        return this.text;
    }

    /**
     * @return a copy of the generated tokens
     */
    public int @NotNull [] getTokens() {
        return this.tokens.clone();
    }

    /**
     * @return why the generation stopped
     */
    public @NotNull StopReason getStopReason() {
        return this.stopReason;
    }

    /**
     * @return the number of tokens of the prompt
     */
    public int getPromptTokenCount() {
        return this.promptTokenCount;
    }

    /**
     * @return the time of decoding the prompts of the request
     */
    public @NotNull Duration getPromptDuration() {
        return this.promptDuration;
    }

    /**
     * @return the number of generated tokens
     */
    public int getGeneratedTokenCount() {
        return this.tokens.length;
    }

    /**
     * @return the time from the end of the prompts to the stop of this sequence
     */
    public @NotNull Duration getGenerationDuration() {
        return this.generationDuration;
    }

    /**
     * Returns a string representation of the object.
     *
     * @return a string representation of the object
     */
    @Override
    public @NotNull String toString() {
        return this.text;
    }
}
