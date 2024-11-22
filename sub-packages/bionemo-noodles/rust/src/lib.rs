use memmap2::Mmap;
use noodles_fasta::{self as fasta, fai};
use pyo3::prelude::*;
use std::fs::File;
use std::io;
use std::path::Path;

#[pyclass]
#[derive(Clone)]
struct PyRecord {
    name: String,
    length: u64,
    offset: u64,
    line_bases: u64,
    line_width: u64,
}

#[pymethods]
impl PyRecord {
    #[getter]
    fn name(&self) -> &str {
        &self.name
    }

    #[getter]
    fn length(&self) -> u64 {
        self.length
    }

    #[getter]
    fn offset(&self) -> u64 {
        self.offset
    }

    #[getter]
    fn line_bases(&self) -> u64 {
        self.line_bases
    }

    #[getter]
    fn line_width(&self) -> u64 {
        self.line_width
    }
    fn __str__(&self) -> String {
        format!(
            "PyRecord(name={}, length={}, offset={}, line_bases={}, line_width={})",
            self.name, self.length, self.offset, self.line_bases, self.line_width
        )
    }

    fn __repr__(&self) -> String {
        format!(
            "<PyRecord name='{}' length={} offset={} line_bases={} line_width={}>",
            self.name, self.length, self.offset, self.line_bases, self.line_width
        )
    }
}

impl From<&fai::Record> for PyRecord {
    fn from(record: &fai::Record) -> Self {
        Self {
            name: String::from_utf8_lossy(record.name()).to_string(),
            length: record.length(),
            offset: record.offset(),
            line_bases: record.line_bases(),
            line_width: record.line_width(),
        }
    }
}

fn fai_record_end_offset(record: &fai::Record) -> usize {
    // gets the offset for the end of the record, as its not available in the index.

    let length = record.length() - 1;
    let num_full_lines = length / record.line_bases();
    let num_bases_remain = length % record.line_bases();

    let bytes_to_last_line_in_record = num_full_lines * record.line_width();

    let bytes_to_end = bytes_to_last_line_in_record + num_bases_remain;

    return (record.offset() + bytes_to_end) as usize;
}

fn query_end_offset(
    record: &fai::Record,
    interval: &noodles_core::region::Interval,
) -> io::Result<usize> {
    // This is lifted from how we compute offset for start position, should be the same.
    let end = interval
        .end() // Extract the end position
        .map(|position| usize::from(position) - 1)
        .unwrap_or_default(); // Default to 0 if unbounded

    // TODO: technically a region with no end is valid, but we pretend its not!
    // subtract 1 to get back to zero based indexing.
    let end = u64::try_from(end).map_err(|e| io::Error::new(io::ErrorKind::InvalidInput, e))?;

    let pos = record.offset() // Start of the contig in bytes
        + end / record.line_bases() * record.line_width() // Full lines before `end`
        + end % record.line_bases(); // Byte offset within the last line

    Ok(pos as usize)
}

fn read_sequence_mmap(index: &fai::Index, reader: &Mmap, region_str: &str) -> io::Result<Vec<u8>> {
    let region: noodles_core::region::Region = region_str.parse().map_err(|e| {
        pyo3::exceptions::PyValueError::new_err(format!("{} Invalid region: {}", e, region_str))
    })?;

    // byte offset for the start of this contig + sequence.
    let start = index.query(&region)?;

    // index record for this contig.
    let record = index
        .as_ref()
        .iter()
        .find(|record| record.name() == region.name())
        .ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("invalid reference sequence name: {}", region.name(),),
            )
        })?;

    // byte offset for the end of this query
    let mut end = query_end_offset(record, &region.interval())?;
    // byte offset for the end of this record, we want to take the smaller of these two, as sometimes we can have silly queries like chr1:1-9999999999999999
    let rec_end = fai_record_end_offset(record);
    end = end.min(fai_record_end_offset(record));

    // call out to our reader and populate the result.
    let mut result = vec![];
    let _ = read_sequence_limit(
        reader,
        start as usize,
        end as usize, // last offset for the sequence
        record.line_bases() as usize,
        record.line_width() as usize,
        record.offset() as usize,
        &mut result,
    );
    return Ok(result);
}

#[pyclass]
struct IndexedMmapFastaReader {
    mmap_reader: memmap2::Mmap,
    index: fai::Index,
}

#[pymethods]
impl IndexedMmapFastaReader {
    #[new]
    fn new(fasta_path: &str) -> PyResult<Self> {
        let fai_path = fasta_path.to_string() + ".fai";
        let fai_path = Path::new(&fai_path); // Convert back to a Path
        let fasta_path = Path::new(fasta_path);

        // Check if the .fai index file exists; if not, create it.
        if !fai_path.exists() {
            // Generate the index by reading the FASTA file
            let index: fai::Index = fasta::io::index(fasta_path).map_err(|e| {
                pyo3::exceptions::PyIOError::new_err(format!("Failed to create index: {}", e))
            })?;

            // Write the index to the .fai file
            let fai_file = File::create(&fai_path).map_err(|e| {
                pyo3::exceptions::PyIOError::new_err(format!("Failed to create .fai file: {}", e))
            })?;

            let mut writer = fai::Writer::new(fai_file);
            writer.write_index(&index).map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!(
                    "Failed to write .fai index: {}",
                    e
                ))
            })?;
        }

        let index: fai::Index = fasta::io::index(fasta_path).map_err(|e| {
            pyo3::exceptions::PyIOError::new_err(format!("Failed to create index: {}", e))
        })?;

        let fd = File::open(fasta_path)?;
        let mmap_reader = unsafe { memmap2::MmapOptions::new().map(&fd) }?;

        Ok(IndexedMmapFastaReader { mmap_reader, index })
    }
    fn records(&self) -> Vec<PyRecord> {
        // return a list of records for the index.
        self.index
            .as_ref()
            .iter()
            .map(|record| PyRecord::from(record))
            .collect()
    }

    fn read_sequence_mmap(&self, region_str: &str) -> PyResult<String> {
        // given a region string, query the value inside the mmap.
        let query_result = &read_sequence_mmap(&self.index, &self.mmap_reader, region_str)?;
        let result = String::from_utf8_lossy(query_result).into_owned();
        return Ok(result);
    }
}

fn bases_remaining_in_first_line_read(
    region_start: usize,
    start: usize,
    line_bases: usize,
    line_width: usize,
) -> usize {
    // Compute the number of bytes from start to the end of the line, half interval.
    //      this means the returned position will the byte offset of a newline.

    let lines_to_start = (start - region_start) / line_width;
    let current_line_start = (line_width * lines_to_start) + region_start;
    let bases_we_skip = start - current_line_start;
    let bases_left_in_line = line_bases - bases_we_skip;

    return bases_left_in_line;
}

fn read_sequence_limit(
    mmap: &Mmap,         // Memory-mapped file
    start: usize,        // Start position in the file (from the index)
    end: usize,          // bases to read
    line_bases: usize,   // Number of bases per line (from the `.fai` index)
    line_width: usize,   // Number of bases per line (from the `.fai` index)
    region_start: usize, // position where the region starts so we can determine the line starts.
    buf: &mut Vec<u8>,   // Buffer to store the sequence
) -> io::Result<usize> {
    // Reads all of the nucleotides from the `start` offset to the `end` offset.
    //   The approach roughly goes like this:
    //       1) read as many bytes as we can until the first newline. This is done analytically so we can make batch reads.
    //       2) read as many complete lines as we can, skipping newlines. Again this is done analytically.
    //       3) read any remaining nucleotides.

    let mut position = start;

    // if we are in the middle of a line, figure out how far to the end
    let mut first_read_to_end =
        position + bases_remaining_in_first_line_read(region_start, start, line_bases, line_width);

    // Handle the special case where we are a subset of a line
    if first_read_to_end > end {
        buf.extend_from_slice(&mmap[position..end + 1]);
        return Ok(1);
    } else {
        // otherwise, read to the end of the line
        buf.extend_from_slice(&mmap[position..first_read_to_end]);
        let bytes_read = first_read_to_end - position;
        position = position + bytes_read + (line_width - line_bases);
    }

    // figure out how many full lines are left.
    let full_lines_to_read = (end - position) / line_width;
    let mut full_lines_read: usize = 0;

    // read as many full lines as we can
    while full_lines_read < full_lines_to_read {
        buf.extend_from_slice(&mmap[position..position + line_bases]);
        full_lines_read += 1;
        position += line_width;
    }

    // if there are any bytes left, read them.
    let remaining_bytes = (end + 1) - position;
    buf.extend_from_slice(&mmap[position..position + remaining_bytes]);
    Ok(full_lines_read)
}

#[pymodule]
fn noodles_fasta_wrapper(_: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<IndexedMmapFastaReader>()?;
    Ok(())
}

#[test]
fn test_query_end_offset() {
    // tests a single row, end of line position
    let record = fai::Record::new("chr1", 12, 6, 12, 13);

    let region_str = "chr1:1-12";
    let region: noodles_core::region::Region = region_str.parse().unwrap();

    let result = query_end_offset(&record, &region.interval()).unwrap();
    // 01 02 03 04 05
    // 06 07 08 09 10 11 12 13 14 15 16 [17] 18
    assert_eq!(result, 17);

    let record = fai::Record::new("chr1", 24, 6, 12, 13);
    let region_str = "chr1:1-24";
    let region: noodles_core::region::Region = region_str.parse().unwrap();

    let result = query_end_offset(&record, &region.interval()).unwrap();
    // 01 02 03 04 05
    // 06 07 08 09 10 11 12 13 14 15 16  17  18
    // 19 20 21 22 23 24 25 26 27 28 29 [30] 31
    assert_eq!(result, 30);

    // tests a three row, beginning of line position
    let record = fai::Record::new("chr1", 25, 6, 12, 13);
    let region_str = "chr1:1-25";
    let region: noodles_core::region::Region = region_str.parse().unwrap();

    let result = query_end_offset(&record, &region.interval()).unwrap();
    //  01 02 03 04 05
    //  06 07 08 09 10 11 12 13 14 15 16 17 18
    //  19 20 21 22 23 24 25 26 27 28 29 30 31
    // [32] 33
    assert_eq!(result, 32);

    // tests a random position within a row.
    let region_str = "chr1:1-6";
    let region: noodles_core::region::Region = region_str.parse().unwrap();

    let result = query_end_offset(&record, &region.interval()).unwrap();
    // 01 02 03 04 05
    // 06 07 08 09 10 [11] 12 13 14 15 16 17 18
    assert_eq!(result, 11);
}

#[test]
fn test_fai_record_end_offset() {
    // tests a single row, end of line position
    let record = fai::Record::new("chr1", 12, 6, 12, 13);

    // expect 17 because offset is 6, 12 characters to read, this is the offset OF THE LAST CHAR, it IS NOT a bound (e.g its inclusive)
    let result = fai_record_end_offset(&record);
    // 01 02 03 04 05
    // 06 07 08 09 10 11 12 13 14 15 16 [17] 18
    // 19 20 21 22 23 24 25 26 27 28 29  30  31
    assert_eq!(result, 17);

    // tests a two row, end of line position
    let record = fai::Record::new("chr1", 24, 6, 12, 13);
    let result = fai_record_end_offset(&record);
    // 01 02 03 04 05
    // 06 07 08 09 10 11 12 13 14 15 16  17  18
    // 19 20 21 22 23 24 25 26 27 28 29 [30] 31
    assert_eq!(result, 30);

    // tests a three row, beginning of line position
    let record = fai::Record::new("chr1", 25, 6, 12, 13);
    let result = fai_record_end_offset(&record);
    //  01 02 03 04 05
    //  06 07 08 09 10 11 12 13 14 15 16 17 18
    //  19 20 21 22 23 24 25 26 27 28 29 30 31
    // [32] 33
    assert_eq!(result, 32);

    // tests a two row, middle of line position
    let record = fai::Record::new("chr1", 20, 6, 12, 13);
    let result = fai_record_end_offset(&record);
    //  01 02 03 04 05
    //  06 07 08 09 10 11 12  13  14 15 16 17 18
    //  19 20 21 22 23 24 25 [26] 27 28 29 30 31
    assert_eq!(result, 26);

    let record = fai::Record::new("chr1", 12, 6, 12, 13);
}

#[test]
fn test_bases_remaining_in_first_line_read() {
    // >seq1
    // ACGTACACGTAC
    // ACGTACGTACGT

    //  01 02 03 04 05
    //  06 07 08  09  10 11 12 13 14 15 16 17 18
    //  19 20 21 [22] 23 24 25 26 27 28 29 30 31
    //
    // region_start = 6
    // start = 22 (16 in base space)
    // line_width = 13
    // line_bases = 12

    // tests a three row, beginning of line position
    let record = fai::Record::new("chr1", 25, 6, 12, 13);
    let start = 22;
    let result = bases_remaining_in_first_line_read(
        record.offset() as usize,
        start,
        record.line_bases() as usize,
        record.line_width() as usize,
    );
    assert_eq!(result, 9);

    let start = 6;
    // mem[6:6+12]
    // this is the null case, where first base is the first character
    let result = bases_remaining_in_first_line_read(
        record.offset() as usize,
        start,
        record.line_bases() as usize,
        record.line_width() as usize,
    );
    assert_eq!(result, record.line_bases() as usize); // should be equal to line_bases since we need to read the whole line.

    // now we are at the very last character
    let start = 17;
    let result = bases_remaining_in_first_line_read(
        record.offset() as usize,
        start,
        record.line_bases() as usize,
        record.line_width() as usize,
    );
    // expect the last position, so the read will be just 1!
    assert_eq!(result, 1);
}
