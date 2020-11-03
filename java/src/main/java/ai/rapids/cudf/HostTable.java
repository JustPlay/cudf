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

import java.util.Arrays;

/**
 * Class to represent a collection of HostColumnVector(s)
 * NOTE: The refcount on the columns will be increased once they are passed in
 */
public final class HostTable implements AutoCloseable {
  private final long rows;
  private HostColumnVector[] columns;

   /**
   * HostTable class makes a copy of the array of {@link HostColumnVector}s passed to it. The class
   * will decrease the refcount on itself and all its contents when closed and free resources if refcount is zero
   * @param columns - Array of HostColumnVector(s)
   */
  public HostTable(HostColumnVector... columns) {
    assert columns != null && columns.length > 0 : "HostColumnVectors can't be null or empty";
    rows = columns[0].getRowCount();

    for (HostColumnVector cv : columns) {
      assert (null != cv) : "ColumnVectors can't be null";
      assert (rows == cv.getRowCount()) : "All columns should have the same number of " + "rows " + cv.getType();
    }

    // Since Arrays are mutable objects make a copy
    this.columns = new HostColumnVector[columns.length];
    for (int i = 0; i < columns.length; i++) {
      this.columns[i] = columns[i];
      columns[i].incRefCount();
    }
  }

  /**
   * Provides a faster way to get access to the columns. Ownership of the table is not transferred by this method
   */
  public HostColumnVector[] getColumns() {
    return columns;
  }

  /**
   * Return the {@link HostColumnVector} at the specified index. If you want to keep a reference to
   * the column around past the life time of the table, you will need to increment the reference
   * count on the column yourself.
   */
  public HostColumnVector getColumn(int index) {
    assert index < columns.length;
    return columns[index];
  }

  public final long getRowCount() {
    return rows;
  }

  public final int getNumberOfColumns() {
    return columns.length;
  }

  @Override
  public void close() {
    if (columns != null) {
      for (int i = 0; i < columns.length; i++) {
        columns[i].close();
        columns[i] = null;
      }
      columns = null;
    }
  }

  @Override
  public String toString() {
    return "HostTable{" +
        "columns=" + Arrays.toString(columns) +
        ", rows=" + rows +
        '}';
  }

  /**
   * Returns the Host memory buffer size.
   */
  public long getHostMemorySize() {
    long total = 0;
    for (HostColumnVector cv: columns) {
      total += cv.getHostMemorySize();
    }
    return total;
  }
}
