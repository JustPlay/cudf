/*
 * Copyright (c) 2019-2020, NVIDIA CORPORATION.
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

package ai.rapids.cudf;

/**
 * The host version of ContiguousTable (java/src/main/java/ai/rapids/cudf/ContiguousTable.java)
 */
public final class ContiguousHostTable implements AutoCloseable {
  private HostTable table;
  private HostMemoryBuffer buffer;

  ContiguousHostTable(HostTable table, HostMemoryBuffer buffer) {
    this.table = table;
    this.buffer = buffer;
  }

  public HostTable getTable() {
    return table;
  }

  public HostMemoryBuffer getBuffer() {
    return buffer;
  }

  @Override
  public void close() {
    if (table != null) {
      table.close();
      table = null;
    }

    if (buffer != null) {
      buffer.close();
      buffer = null;
    }
  }
}
