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

/**
 * The reasons why the generation of a sequence stops.
 */
public enum StopReason {
    /**
     * The model generated an end-of-generation token, for example the end-of-turn token of a chat format.
     */
    END_OF_GENERATION,

    /**
     * The sequence reached the maximum number of generated tokens.
     */
    MAXIMUM_TOKENS,

    /**
     * The sequence filled its part of the context memory.
     */
    CONTEXT_FULL
}
