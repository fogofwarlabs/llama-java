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
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

/**
 * Checks the GPU layers and devices of the model options.
 */
final class ModelOptionsTest {
    @Test
    void set_devices_should_reject_the_cpu() {
        LlamaDevice cpu = LlamaRuntime.getInstance().getDevices().stream()
                                      .filter(device -> device.getType() == LlamaDeviceType.CPU)
                                      .findFirst()
                                      .orElseThrow();
        ModelOptions options = new ModelOptions();

        Assertions.assertThrows(IllegalArgumentException.class, () -> options.setDevices(List.of(cpu)));
    }

    @Test
    void set_gpu_layers_should_reject_negative_counts_other_than_all_layers() {
        ModelOptions options = new ModelOptions();

        Assertions.assertEquals(ModelOptions.ALL_LAYERS, options.getGpuLayers());
        Assertions.assertEquals(0, options.setGpuLayers(0).getGpuLayers());
        Assertions.assertThrows(IllegalArgumentException.class, () -> options.setGpuLayers(-2));
    }
}
