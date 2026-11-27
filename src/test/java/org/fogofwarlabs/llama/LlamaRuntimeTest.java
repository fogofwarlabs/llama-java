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
import java.util.List;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

/**
 * Loads the native libraries of this machine and lists the devices of the loaded backends.
 */
final class LlamaRuntimeTest {
    @Test
    void get_instance_should_load_the_libraries_and_the_cpu_backend() {
        LlamaRuntime runtime = LlamaRuntime.getInstance();
        List<LlamaDevice> devices = runtime.getDevices();

        System.out.println("llama.cpp " + runtime.getBuild() + " " + runtime.getPlatform() + " from " + runtime.getLibraryDirectory());
        System.out.println(runtime.getSystemInfo());

        for (LlamaDevice device : devices) {
            System.out.println(device + ", backend " + device.getBackendName() + ", " + device.getFreeMemory() + " of " +
                               device.getTotalMemory() + " bytes free");
        }

        Assertions.assertSame(runtime, LlamaRuntime.getInstance());
        Assertions.assertTrue(runtime.getBuild().matches("b\\d+"), runtime.getBuild());
        Assertions.assertTrue(Files.isDirectory(runtime.getLibraryDirectory()));
        Assertions.assertFalse(runtime.getSystemInfo().isEmpty());
        Assertions.assertTrue(devices.stream().anyMatch(device -> device.getType() == LlamaDeviceType.CPU));
        Assertions.assertTrue(devices.stream().allMatch(device -> device.getTotalMemory() >= device.getFreeMemory()));
        Assertions.assertTrue(devices.containsAll(runtime.getGpuDevices()));
    }
}
