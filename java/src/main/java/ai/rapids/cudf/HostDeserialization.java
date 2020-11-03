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

import java.io.BufferedOutputStream;
import java.io.Closeable;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.EOFException;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.ByteBuffer;
import java.util.Optional;

/**
 * Deserialize CUDF tables on HOST (not using GPU)
 * NOTE: Most of the code are copied from java/src/main/java/ai/rapids/cudf/JCudfSerialization.java
 */
public class HostDeserialization {
  /**
   * Magic number "CUDF" in ASCII, which is 1178883395 if read in LE from big endian, which is
   * too large for any reasonable metadata for arrow, so we should probably be okay detecting
   * this, and switching back/forth at a later time.
   * 
   * NOTE: copied from JCudfSerialization.java WITHOUT modification
   */
  private static final int SER_FORMAT_MAGIC_NUMBER = 0x43554446;
  private static final short VERSION_NUMBER = 0x0000;

  private static final class ColumnOffsets {
    private final long validity;
    private final long validityLen;
    private final long offsets;
    private final long offsetsLen;
    private final long data;
    private final long dataLen;

    public ColumnOffsets(long validity, long validityLen,
                         long offsets, long offsetsLen,
                         long data, long dataLen) {
      this.validity = validity;
      this.validityLen = validityLen;
      this.offsets = offsets;
      this.offsetsLen = offsetsLen;
      this.data = data;
      this.dataLen = dataLen;
    }
  }

  /**
   * Holds the metadata about a serialized table. If this is being read from a stream
   * isInitialized will return true if the metadata was read correctly from the stream.
   * It will return false if an EOF was encountered at the beginning indicating that
   * there was no data to be read.
   * 
   * NOTE: copied from JCudfSerialization.java WITHOUT modification
   */
  public static final class SerializedTableHeader {
    private int numColumns;
    int numRows;

    private DType[] types;
    private long[] nullCounts;
    long dataLen;

    private boolean initialized = false;
    private boolean dataRead = false;

    public SerializedTableHeader(DataInputStream din) throws IOException {
      readFrom(din);
    }

    SerializedTableHeader(int numRows, DType[] types, long[] nullCounts, long dataLen) {
      this.numRows = numRows;
      if (types != null) {
        numColumns = types.length;
      } else {
        numColumns = 0;
      }
      this.types = types;
      this.nullCounts = nullCounts;
      this.dataLen = dataLen;
      initialized = true;
      dataRead = true;
    }

    public DType getColumnType(int columnIndex) {
      return types[columnIndex];
    }

    /**
     * Set to true once data is successfully read from a stream by readTableIntoBuffer.
     * @return true if data was read, else false.
     */
    public boolean wasDataRead() {
      return dataRead;
    }

    /**
     * Returns the size of a buffer needed to read data into the stream.
     */
    public long getDataLen() {
      return dataLen;
    }

    /**
     * Returns the number of rows stored in this table.
     */
    public int getNumRows() {
      return numRows;
    }

    /**
     * Returns the number of columns stored in this table
     */
    public int getNumColumns() {
      return numColumns;
    }

    /**
     * Returns true if the metadata for this table was read, else false indicating an EOF was
     * encountered.
     */
    public boolean wasInitialized() {
      return initialized;
    }

    private void readFrom(DataInputStream din) throws IOException {
      try {
        int num = din.readInt();
        if (num != SER_FORMAT_MAGIC_NUMBER) {
          throw new IllegalStateException("THIS DOES NOT LOOK LIKE CUDF SERIALIZED DATA. " +
              "Expected magic number " + SER_FORMAT_MAGIC_NUMBER + " Found " + num);
        }
      } catch (EOFException e) {
        // If we get an EOF at the very beginning don't treat it as an error because we may
        // have finished reading everything...
        return;
      }
      short version = din.readShort();
      if (version != VERSION_NUMBER) {
        throw new IllegalStateException("READING THE WRONG SERIALIZATION FORMAT VERSION FOUND "
            + version + " EXPECTED " + VERSION_NUMBER);
      }
      numColumns = din.readInt();
      numRows = din.readInt();

      types = new DType[numColumns];
      nullCounts = new long[numColumns];
      for (int i = 0; i < numColumns; i++) {
        types[i] = DType.fromNative(din.readInt(), din.readInt());
        nullCounts[i] = din.readInt();
      }

      dataLen = din.readLong();
      initialized = true;
    }
  }

  // NOTE: copied from JCudfSerialization.java WITHOUT modification
  private static long padFor64byteAlignment(long orig) {
    return ((orig + 63) / 64) * 64;
  }

  // NOTE: copied from JCudfSerialization.java WITHOUT modification
  static ColumnOffsets[] buildIndex(SerializedTableHeader header,
                                    HostMemoryBuffer buffer) {
    long bufferOffset = 0;
    DType[] dataTypes = header.types;
    int numColumns = dataTypes.length;
    long[] nullCounts = header.nullCounts;
    long numRows = header.getNumRows();
    ColumnOffsets[] ret = new ColumnOffsets[numColumns];
    for (int column = 0; column < numColumns; column++) {
      DType type = dataTypes[column];
      long nullCount = nullCounts[column];

      long validity = 0;
      long validityLen = 0;
      long offsets = 0;
      long offsetsLen = 0;
      long data = 0;
      long dataLen = 0;
      if (nullCount > 0) {
        validityLen = padFor64byteAlignment(BitVectorHelper.getValidityLengthInBytes(numRows));
        validity = bufferOffset;
        bufferOffset += validityLen;
      }

      if (type == DType.STRING) {
        if (numRows > 0) {
          offsetsLen = (numRows + 1) * 4;
          offsets = bufferOffset;
          int startStringOffset = buffer.getInt(bufferOffset);
          int endStringOffset = buffer.getInt(bufferOffset + (numRows * 4));
          bufferOffset += padFor64byteAlignment(offsetsLen);

          dataLen = endStringOffset - startStringOffset;
          data = bufferOffset;
          bufferOffset += padFor64byteAlignment(dataLen);
        }
      } else {
        dataLen = type.getSizeInBytes() * numRows;
        data = bufferOffset;
        bufferOffset += padFor64byteAlignment(dataLen);
      }
      ret[column] = new ColumnOffsets(validity, validityLen,
          offsets, offsetsLen,
          data, dataLen);
    }
    return ret;
  }

  // NOTE: copied from JCudfSerialization.java WITH modification
  private static HostTable sliceUpColumnVectors(SerializedTableHeader header, HostMemoryBuffer combinedBufferOnHost) {
    try (NvtxRange range = new NvtxRange("bufferToTable", NvtxColor.PURPLE)) {
      ColumnOffsets[] columnOffsets = buildIndex(header, combinedBufferOnHost);
      DType[] dataTypes = header.types;
      long[] nullCounts = header.nullCounts;
      long numRows = header.getNumRows();
      int numColumns = dataTypes.length;
      HostColumnVector[] vectors = new HostColumnVector[numColumns];
      HostMemoryBuffer validity = null;
      HostMemoryBuffer data = null;
      HostMemoryBuffer offsets = null;
      try {
        for (int column = 0; column < numColumns; column++) {
          DType type = dataTypes[column];
          long nullCount = nullCounts[column];
          ColumnOffsets offsetInfo = columnOffsets[column];

          if (nullCount > 0) {
            validity = combinedBufferOnHost.slice(offsetInfo.validity, offsetInfo.validityLen);
          }

          if (type == DType.STRING) {
            offsets = combinedBufferOnHost.slice(offsetInfo.offsets, offsetInfo.offsetsLen);
          }

          // The vector is possibly full of null strings. This is a rare corner case, but we let
          // data buffer stay null.
          if (offsetInfo.dataLen > 0) {
            data = combinedBufferOnHost.slice(offsetInfo.data, offsetInfo.dataLen);
          }

          vectors[column] = new HostColumnVector(type, numRows, Optional.of(nullCount), data, validity, offsets, null);
          validity = null;
          data = null;
          offsets = null;
        }
        return new HostTable(vectors);
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
        // the refcount had been increased in the ctor of HostTable
        for (HostColumnVector cv: vectors) {
          if (cv != null) {
            cv.close();
          }
        }
      }
    }
  }

  /**
   * After reading a header for a table read the data portion into a host side buffer.
   * @param in the stream to read the data from.
   * @param header the header that finished just moments ago.
   * @param buffer the buffer to write the data into.  If there is not enough room to store
   *               the data in buffer it will not be read and header will still have dataRead
   *               set to false.
   * @throws IOException
   * 
   * NOTE: copied from JCudfSerialization.java WITHOUT modification
   */
  public static void readTableIntoBuffer(InputStream in,
                                         SerializedTableHeader header,
                                         HostMemoryBuffer buffer) throws IOException {
    if (header.initialized &&
        (buffer.length >= header.dataLen)) {
      try (NvtxRange range = new NvtxRange("Read Data", NvtxColor.RED)) {
        buffer.copyFromStream(0, in, header.dataLen);
      }
      header.dataRead = true;
    }
  }

  // NOTE: copied from JCudfSerialization.java WITH modification
  public static TableAndRowCountPair readTableFrom(SerializedTableHeader header, HostMemoryBuffer hostBuffer) {
    ContiguousHostTable contigHostTable = null;                            
    if (header.getNumColumns() > 0) {
      HostTable hostTable = sliceUpColumnVectors(header, hostBuffer);
      contigHostTable = new ContiguousHostTable(hostTable, hostBuffer);
    }

    return new TableAndRowCountPair(header.numRows, contigHostTable);
  }

  /**
   * Read a serialize table from the given InputStream.
   * @param in the stream to read the table data from.
   * @return the deserialized table in host memory, or null if the stream has no table to read
   * from, an end of the stream at the very beginning.
   * @throws IOException on any error.
   * @throws EOFException if the data stream ended unexpectedly in the middle of processing.
   * 
   * NOTE: copied from JCudfSerialization.java WITH modification
   */
  public static TableAndRowCountPair readTableFrom(InputStream in) throws IOException {
    DataInputStream din;
    if (in instanceof DataInputStream) {
      din = (DataInputStream)in;
    } else {
      din = new DataInputStream(in);
    }

    SerializedTableHeader header = new SerializedTableHeader(din);
    if (!header.initialized) {
      return new TableAndRowCountPair(0, null);
    }

    try (HostMemoryBuffer hostBuffer = HostMemoryBuffer.allocate(header.dataLen)) {
      if (header.dataLen > 0) {
        readTableIntoBuffer(din, header, hostBuffer);
      }
      return readTableFrom(header, hostBuffer);
    }
  }

  /** 
   * Holds the result of deserializing a table. 
   * 
   * NOTE: copied from JCudfSerialization.java WITH modification
   */
  public static final class TableAndRowCountPair implements Closeable {
    private final int numRows;
    private final ContiguousHostTable contigHostTable;

    public TableAndRowCountPair(int numRows, ContiguousHostTable table) {
      this.numRows = numRows;
      this.contigHostTable = table;
    }

    @Override
    public void close() {
      if (contigHostTable != null) {
        contigHostTable.close();
      }
    }

    /** Get the number of rows that were deserialized. */
    public int getNumRows() {
          return numRows;
      }

    /**
     * Get the Table that was deserialized or null if there was no data
     * (e.g.: rows without columns).
     * <p>NOTE: Ownership of the table is not transferred by this method.
     * The table is still owned by this instance and will be closed when this
     * instance is closed.
     */
    public HostTable getTable() {
      if (contigHostTable != null) {
        return contigHostTable.getTable();
      }
      return null;
    }

    /**
     * Get the ContiguousTable that was deserialized or null if there was no
     * data (e.g.: rows without columns).
     * <p>NOTE: Ownership of the contiguous table is not transferred by this
     * method. The contiguous table is still owned by this instance and will
     * be closed when this instance is closed.
     */
    public ContiguousHostTable getContiguousTable() {
      return contigHostTable;
    }
  }
}
