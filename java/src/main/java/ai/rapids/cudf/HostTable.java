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
import java.util.Optional;

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

  /**
   * copy the host-side HostTable to gpu-side Table
   * 
   * @param hostBuffer - a HostMemoryBuffer
   * @param columns - Array of HostColumnVector(s)
   * 
   * @return a Table object
   * 
   * NOTE: @hostBuffer and @columns are from a ContiguousHostTable, 
   *       all the HostColumnVector(s) must share the same single underlying HostMemoryBuffer
   */
  public static Table copyToDevice(HostMemoryBuffer hostBuffer, HostColumnVector[] columns) {
    Table table = null;
    DeviceMemoryBuffer devBuffer = null;
    try (NvtxRange range = new NvtxRange("HostBuffers->GpuTable", NvtxColor.PURPLE)) {
      // allocate a single devide-side buffer, and copy data from @hostBuffer
      devBuffer = DeviceMemoryBuffer.allocate(hostBuffer.getLength());
      devBuffer.copyFromHostBuffer(hostBuffer);

      // build column layout based-on the host-side buffer and then slice the device-side buffer into mulltiple columns, 
      // finally warp them into a Table
      ColumnInfo[] columnInfo = buildLayout(hostBuffer, columns);
      table = buildTableBasedonLayoutInfo(devBuffer, columnInfo);
    } finally {
      if (devBuffer != null) {
        devBuffer.close();
      }
    }

    return table;
  }

  private static final class ColumnInfo {
    private final DType type;
    private final long nullCount;
    private final long numRows;
    private final long validity;    // the validity buffer's offset in the underlying buffer
    private final long validityLen; // the validity buffer's length
    private final long offsets;
    private final long offsetsLen;
    private final long data;
    private final long dataLen;

    public ColumnInfo(DType type, long nullCount, long numRows,
                      long validity, long validityLen,
                      long offsets, long offsetsLen,
                      long data, long dataLen) {
      this.type = type;
      this.nullCount = nullCount;
      this.numRows = numRows;
      this.validity = validity;
      this.validityLen = validityLen;
      this.offsets = offsets;
      this.offsetsLen = offsetsLen;
      this.data = data;
      this.dataLen = dataLen;
    }

    // NOTE: this class does not take into account the `children`, so we do not support nested type, i.e. LIST, STRUCT
    //       this version's cuDF itself does not support nested type too, see JCudfSerialization.java:buildIndex() for more detailed info
  }

  private static ColumnInfo[] buildLayout(HostMemoryBuffer hostBuffer, HostColumnVector... columns) {
    long bufferAddress = hostBuffer.getAddress();
    int numColumns = columns.length;
    
    ColumnInfo[] ci = new ColumnInfo[numColumns];
    for (int i = 0; i < numColumns; ++i) {
      DType type = columns[i].getDataType();
      long nullCount = columns[i].getNullCount();
      long numRows = columns[i].getNumRows();
      long validity = 0;
      long validityLen = 0;
      long offsets = 0;
      long offsetsLen = 0;
      long data = 0;
      long dataLen = 0;

      HostMemoryBuffer buf = null;
     
      buf = columns[i].getValidityBuffer();
      if (buf != null) {
        validity = buf.getAddress() - bufferAddress;
        validityLen = buf.getLength();
      }

      buf = columns[i].getOffsetBuffer();
      if (buf != null) {
        offsets = buf.getAddress() - bufferAddress;
        offsetsLen = buf.getLength();
      }

      buf = columns[i].getDataBuffer(); 
      if (buf != null) {
        data = buf.getAddress() - bufferAddress;
        dataLen = buf.getLength();
      }

      ci[i] = new ColumnInfo(type, nullCount, numRows, validity, validityLen, offsets, offsetsLen, data, dataLen);
    }

    return ci;
  }

  private static Table buildTableBasedonLayoutInfo(DeviceMemoryBuffer devBuffer, ColumnInfo... columns) {
    int numColumns = columns.length;
    long numRows = columns[0].numRows;

    ColumnVector[] vectors = new ColumnVector[numColumns];
    DeviceMemoryBuffer validity = null;
    DeviceMemoryBuffer data = null;
    DeviceMemoryBuffer offsets = null;
    try {
      for (int i = 0; i < numColumns; i++) {
        ColumnInfo ci = columns[i];
        DType type = ci.type;
        long nullCount = ci.nullCount;

        if (ci.validityLen > 0) {
          validity = devBuffer.slice(ci.validity, ci.validityLen);
        }

        if (ci.offsetsLen > 0) {
          offsets = devBuffer.slice(ci.offsets, ci.offsetsLen);
        }

        if (ci.dataLen > 0) {
          data = devBuffer.slice(ci.data, ci.dataLen);
        }

        vectors[i] = new ColumnVector(type, numRows, Optional.of(nullCount), data, validity, offsets);
        validity = null;
        data = null;
        offsets = null;
      }
      return new Table(vectors);
    } finally {
      if (validity != null) {
        validity.close();
      }

      if (data != null) {
        data.close();
      }

      if (offsets != null) {
        offsets.close();
      }

      for (ColumnVector cv: vectors) {
        if (cv != null) {
          cv.close();
        }
      }
    }
  }
}
