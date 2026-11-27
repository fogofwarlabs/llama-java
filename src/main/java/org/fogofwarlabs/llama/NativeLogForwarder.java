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

import static org.fogofwarlabs.llama.bindings.llama_h.GGML_LOG_LEVEL_CONT;
import static org.fogofwarlabs.llama.bindings.llama_h.GGML_LOG_LEVEL_ERROR;
import static org.fogofwarlabs.llama.bindings.llama_h.GGML_LOG_LEVEL_INFO;
import static org.fogofwarlabs.llama.bindings.llama_h.GGML_LOG_LEVEL_WARN;
import static org.fogofwarlabs.llama.bindings.llama_h.llama_log_set;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import org.fogofwarlabs.llama.bindings.ggml_log_callback;
import org.jetbrains.annotations.NotNull;

/**
 * Receives the log messages of llama.cpp and ggml and writes each line to the {@link System.Logger} {@value #LOGGER_NAME}.
 * <p>
 * Errors and warnings keep their level, information messages are written at {@link System.Logger.Level#DEBUG} and debug messages at
 * {@link System.Logger.Level#TRACE}. Message pieces are joined until a line ends.
 */
final class NativeLogForwarder {
    /**
     * The name of the logger that receives the native log.
     */
    static final String LOGGER_NAME = "org.fogofwarlabs.llama";

    private static final char LINE_END = '\n';

    private static @NotNull System.Logger.Level toLoggerLevel(int level) {
        if (level == GGML_LOG_LEVEL_ERROR()) {
            return System.Logger.Level.ERROR;
        }

        if (level == GGML_LOG_LEVEL_WARN()) {
            return System.Logger.Level.WARNING;
        }

        if (level == GGML_LOG_LEVEL_INFO()) {
            return System.Logger.Level.DEBUG;
        }

        return System.Logger.Level.TRACE;
    }

    private final @NotNull System.Logger logger;

    private final @NotNull StringBuilder line;
    private @NotNull System.Logger.Level lineLevel;

    /**
     * Creates a {@link NativeLogForwarder} instance that writes to the logger {@value #LOGGER_NAME}.
     */
    NativeLogForwarder() {
        this.logger = System.getLogger(LOGGER_NAME);

        this.line      = new StringBuilder();
        this.lineLevel = System.Logger.Level.DEBUG;
    }

    /**
     * Installs this forwarder as the log callback of llama.cpp and ggml. The callback stays valid until the process ends.
     */
    void install() {
        MemorySegment callback = ggml_log_callback.allocate(this::receive, Arena.global());

        llama_log_set(callback, MemorySegment.NULL);
    }

    private synchronized void receive(int level, @NotNull MemorySegment text, @NotNull MemorySegment userData) {
        // An exception must not reach the native caller.
        try {
            if (level != GGML_LOG_LEVEL_CONT()) {
                this.lineLevel = toLoggerLevel(level);
            }

            if (!text.equals(MemorySegment.NULL)) {
                this.line.append(text.getString(0));
            }

            int end;

            while ((end = this.line.indexOf(String.valueOf(LINE_END))) >= 0) {
                String message = this.line.substring(0, end).stripTrailing();

                this.line.delete(0, end + 1);

                if (!message.isEmpty()) {
                    this.logger.log(this.lineLevel, message);
                }
            }
        } catch (Throwable ignored) {
            this.line.setLength(0);
        }
    }
}
