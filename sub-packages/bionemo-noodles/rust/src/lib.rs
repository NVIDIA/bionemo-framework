use memmap2::Mmap;
use noodles_fasta::{self as fasta, fai};
use pyo3::prelude::*;
use pyo3::types::PyType;
use std::fs::File;
use std::io;
use std::path::Path;
use std::path::PathBuf;

/// Python wrapper around the faidx Record struct.
/// Fields:
/// - name: name of the record, corresponds to a sequence id in the indexed fasta file.
/// - length: length of the record, number of bases/nucleotides/characters in the record, including Ns.
/// - offset: offset of the record's first base/nucleotide/character in bytes, from the start of the file.
/// - line_bases: number of bases per line in the fasta file
/// - line_width: number of bytes per line in the fasta file, including newlines, return carriages, etc.
#[pyclass]
#[derive(Clone)]
struct PyFaidxRecord {
    name: String,
    length: u64,
    offset: u64,
    line_bases: u64,
    line_width: u64,
}

#[pymethods]
impl PyFaidxRecord {
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
            "PyFaidxRecord(name={}, length={}, offset={}, line_bases={}, line_width={})",
            self.name, self.length, self.offset, self.line_bases, self.line_width
        )
    }

    fn __repr__(&self) -> String {
        format!(
            "<PyFaidxRecord name={} length={} offset={} line_bases={} line_width={}>",
            self.name, self.length, self.offset, self.line_bases, self.line_width
        )
    }
}

impl From<&fai::Record> for PyFaidxRecord {
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

#[pyclass]
struct PyIndexedMmapFastaReader {
    inner: IndexedMmapFastaReader,
}

#[pymethods]
impl PyIndexedMmapFastaReader {
    #[new]
    #[pyo3(signature = (fasta_path, ignore_existing_fai = true))]
    fn new(fasta_path: &str, ignore_existing_fai: bool) -> PyResult<Self> {
        match IndexedMmapFastaReader::new(fasta_path, ignore_existing_fai) {
            Ok(inner) => Ok(Self { inner }),
            Err(e) => {
                let py_err = match e.kind() {
                    std::io::ErrorKind::NotFound => {
                        pyo3::exceptions::PyFileNotFoundError::new_err(format!("{}", e))
                    }
                    std::io::ErrorKind::PermissionDenied => {
                        pyo3::exceptions::PyPermissionError::new_err(format!("{}", e))
                    }
                    _ => pyo3::exceptions::PyRuntimeError::new_err(format!("{}", e)),
                };
                Err(py_err)
            }
        }
    }
    /// Create a new IndexedMmapFastaReader from a fasta file and a fai file explicitly.
    #[classmethod]
    fn from_fasta_and_faidx(
        _cls: &PyType,
        fasta_filename: &str,
        fasta_fai_filename: &str,
    ) -> PyResult<Self> {
        match IndexedMmapFastaReader::from_fasta_and_faidx(fasta_filename, fasta_fai_filename) {
            Ok(inner) => Ok(Self { inner }),
            Err(e) => {
                let py_err = match e.kind() {
                    std::io::ErrorKind::NotFound => {
                        pyo3::exceptions::PyFileNotFoundError::new_err(format!("{}", e))
                    }
                    std::io::ErrorKind::PermissionDenied => {
                        pyo3::exceptions::PyPermissionError::new_err(format!("{}", e))
                    }
                    _ => pyo3::exceptions::PyRuntimeError::new_err(format!("{}", e)),
                };
                Err(py_err)
            }
        }
    }

    #[staticmethod]
    fn create_faidx(fasta_filename: &str) -> PyResult<String> {
        match IndexedMmapFastaReader::create_faidx(fasta_filename) {
            Ok(fai_filename) => Ok(fai_filename),
            Err(e) => {
                let py_err = match e.kind() {
                    std::io::ErrorKind::AlreadyExists => {
                        pyo3::exceptions::PyFileExistsError::new_err(format!("{}", e))
                    }
                    std::io::ErrorKind::PermissionDenied => {
                        pyo3::exceptions::PyPermissionError::new_err(format!("{}", e))
                    }
                    _ => pyo3::exceptions::PyRuntimeError::new_err(format!("{}", e)),
                };
                Err(py_err)
            }
        }
    }

    fn records(&self) -> Vec<PyFaidxRecord> {
        return self
            .inner
            .index
            .as_ref()
            .iter()
            .map(|record| PyFaidxRecord::from(record))
            .collect();
    }
    fn read_sequence_mmap(&self, region_str: &str) -> PyResult<String> {
        self.inner
            .read_sequence_mmap(region_str)
            .map_err(|e| match e.kind() {
                std::io::ErrorKind::InvalidInput => {
                    pyo3::exceptions::PyValueError::new_err(format!("Invalid input: {}", e))
                }
                std::io::ErrorKind::NotFound => {
                    pyo3::exceptions::PyFileNotFoundError::new_err(format!("File not found: {}", e))
                }
                std::io::ErrorKind::PermissionDenied => {
                    pyo3::exceptions::PyPermissionError::new_err(format!(
                        "Permission denied: {}",
                        e
                    ))
                }
                _ => pyo3::exceptions::PyRuntimeError::new_err(format!("Unexpected error: {}", e)),
            })
    }
}

#[pymodule]
fn noodles_fasta_wrapper(_: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyIndexedMmapFastaReader>()?;
    m.add_class::<PyFaidxRecord>()?;
    Ok(())
}

struct IndexedMmapFastaReader {
    mmap_reader: memmap2::Mmap,
    index: fai::Index,
}

impl IndexedMmapFastaReader {
    fn from_fasta_and_faidx(fasta_path: &str, fasta_fai_path: &str) -> std::io::Result<Self> {
        let fasta_path = Path::new(fasta_path);
        if !fasta_path.exists() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("Fasta file {} not found", fasta_path.display()),
            ));
        }
        let index = load_index_from_filename(fasta_fai_path)?;
        let fd = File::open(fasta_path)?;
        let mmap_reader = unsafe { memmap2::MmapOptions::new().map(&fd) }?;
        Ok(IndexedMmapFastaReader { mmap_reader, index })
    }

    fn from_fasta(fasta_path: &str) -> std::io::Result<Self> {
        let fasta_path = Path::new(fasta_path);
        let fd = File::open(fasta_path)?;
        let index: fai::Index = fasta::io::index(fasta_path).map_err(|e| {
            std::io::Error::new(
                std::io::ErrorKind::Other,
                format!(
                    "For fasta file {}, Failed to create index: {}",
                    fasta_path.display(),
                    e
                ),
            )
        })?;
        let mmap_reader = unsafe { memmap2::MmapOptions::new().map(&fd) }?;
        Ok(IndexedMmapFastaReader { mmap_reader, index })
    }

    fn create_faidx(fasta_filename: &str) -> std::io::Result<String> {
        let fasta_path = Path::new(fasta_filename);
        let index: fai::Index = fasta::io::index(fasta_path).map_err(|e| {
            std::io::Error::new(
                std::io::ErrorKind::Other,
                format!(
                    "For fasta file {}, Failed to create index: {}",
                    fasta_path.display(),
                    e
                ),
            )
        })?;

        let fai_filename = fasta_filename.to_string() + ".fai";
        let fai_path = Path::new(&fai_filename); // Convert back to a Path
        if fai_path.exists() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::AlreadyExists,
                format!("Fai file {} already exists", fai_path.display()),
            ));
        }

        let fai_file = File::create(&fai_path).map_err(|e| {
            std::io::Error::new(
                std::io::ErrorKind::Other,
                format!("Failed to create .fai file: {}", e),
            )
        })?;
        let mut writer = fai::Writer::new(fai_file);
        writer.write_index(&index).map_err(|e| {
            std::io::Error::new(
                std::io::ErrorKind::Other,
                format!("Failed to write .fai index: {}", e),
            )
        })?;

        return Ok(fai_filename);
    }

    fn new(fasta_path: &str, ignore_existing_fai: bool) -> std::io::Result<Self> {
        if !ignore_existing_fai {
            // load the .fai files if they exist
            let fasta_fai_path = fasta_path.to_string() + ".fai";
            Self::from_fasta_and_faidx(fasta_path, &fasta_fai_path as &str)
        } else {
            Self::from_fasta(fasta_path)
        }
    }

    fn read_sequence_mmap(&self, region_str: &str) -> std::io::Result<String> {
        // given a region string, query the value inside the mmap.
        let query_result = &read_sequence_mmap(&self.index, &self.mmap_reader, region_str)?;
        let result = String::from_utf8_lossy(query_result).into_owned();
        return Ok(result);
    }
}

/// gets the byte offset for the last base of the record, as its not available in the index.
fn fai_record_end_offset(record: &fai::Record) -> usize {
    let length = record.length() - 1;
    let num_full_lines = length / record.line_bases();
    let num_bases_remain = length % record.line_bases();

    let bytes_to_last_line_in_record = num_full_lines * record.line_width();
    let bytes_to_end = bytes_to_last_line_in_record + num_bases_remain;

    return (record.offset() + bytes_to_end) as usize;
}

/// Given a record and an interval, compute the byte offset for the last byte included in the interval.
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

/// Given an index, a memory-mapped file, and a region string, read the sequence from the file.
///     This function unwraps the region string, clams the query to the final read, and then invokes the read function.
fn read_sequence_mmap(index: &fai::Index, reader: &Mmap, region_str: &str) -> io::Result<Vec<u8>> {
    let region: noodles_core::region::Region = region_str.parse().map_err(|e| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!("{} Invalid region: {}", e, region_str),
        )
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
    end = end.min(fai_record_end_offset(record));

    // call out to our reader and populate the result.
    let mut result = vec![];
    let _ = read_range_from_mmap(
        reader,
        start as usize,
        end as usize, // last offset for the sequence
        &record,
        &mut result,
    );
    return Ok(result);
}

/// Compute the number of bytes from start to the end of the line, half interval.
///      this means the returned position will the byte offset of a newline.
fn bases_remaining_in_first_line_read(
    region_start: usize,
    start: usize,
    line_bases: usize,
    line_width: usize,
) -> usize {
    let lines_to_start = (start - region_start) / line_width;
    let current_line_start = (line_width * lines_to_start) + region_start;
    let bases_we_skip = start - current_line_start;
    let bases_left_in_line = line_bases - bases_we_skip;

    return bases_left_in_line;
}

fn read_range_from_mmap(
    mmap: &Mmap,                // Memory-mapped file
    start: usize,               // Start position in the file (from the index)
    end: usize,                 // bases to read
    index_record: &fai::Record, // index record for the contig
    buf: &mut Vec<u8>,          // Buffer to store the sequence
) -> io::Result<usize> {
    // Reads all of the nucleotides from the `start` offset to the `end` offset.
    //   The approach roughly goes like this:
    //       1) read as many bytes as we can until the first newline. This is done analytically so we can make batch reads.
    //       2) read as many complete lines as we can, skipping newlines. Again this is done analytically.
    //       3) read any remaining nucleotides.

    // some convenient unpacking
    let line_bases: usize = index_record.line_bases() as usize;
    let line_width: usize = index_record.line_width() as usize;
    let region_start: usize = index_record.offset() as usize;

    let mut position = start;

    // if we are in the middle of a line, figure out how far to the end
    let first_read_to_end =
        position + bases_remaining_in_first_line_read(region_start, start, line_bases, line_width);

    // Handle the special case where we are a subset of a line
    if first_read_to_end > end {
        buf.extend_from_slice(&mmap[position..end + 1]);
        return Ok(buf.len());
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
    Ok(buf.len())
}

fn load_index_from_filename(fai_path: &str) -> Result<fai::Index, std::io::Error> {
    let fai_path = PathBuf::from(fai_path);
    let fai_fd = File::open(fai_path)?;
    let mut reader = fai::io::Reader::new(std::io::BufReader::new(fai_fd)); // Wrap the File in a BufReader
    let idx = reader.read_index();
    return idx;
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

#[test]
fn test_invalid_fai_fails() {
    let fai_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")) // Base directory of the project
        .join("tests/bionemo/noodles/data/bad_index.fasta.fai");
    let fai_fd = File::open(&fai_path).unwrap();
    let mut reader = fai::io::Reader::new(std::io::BufReader::new(fai_fd)); // Wrap the File in a BufReader
    let index: Result<fai::Index, _> = reader.read_index();
    assert!(index.is_err());

    // tests our impl to make sures it matches above
    let index = load_index_from_filename(&fai_path.to_str().unwrap());
    assert!(index.is_err());
}

#[test]
fn test_create_from_fasta_faidx_no_fai() {
    IndexedMmapFastaReader::from_fasta_and_faidx(
        "tests/bionemo/noodles/data/sample.fasta",
        "tests/bionemo/noodles/data/sample.fasta.fai",
    )
    .unwrap();
    assert!(IndexedMmapFastaReader::from_fasta_and_faidx(
        "tests/bionemo/noodles/data/sample.fasta",
        "asdfasdfasdf"
    )
    .is_err());
}

#[test]
fn test_valid_fai_is_read() {
    let fai_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")) // Base directory of the project
        .join("tests/bionemo/noodles/data/sample.fasta.fai");
    let fai_fd = File::open(&fai_path).unwrap();
    let mut reader = fai::io::Reader::new(std::io::BufReader::new(fai_fd)); // Wrap the File in a BufReader
    let index = reader.read_index().unwrap();

    let records: Vec<_> = index.as_ref().iter().collect();
    assert_eq!(records.len(), 5);

    // test our implementation to ensure it matches.
    let index = load_index_from_filename(&fai_path.to_str().unwrap()).unwrap();
    let records: Vec<_> = index.as_ref().iter().collect();
    assert_eq!(records.len(), 5);
}

#[test]
fn test_mmap_reads() {
    let fasta_filename = PathBuf::from(env!("CARGO_MANIFEST_DIR")) // Base directory of the project
        .join("tests/bionemo/noodles/data/sample.fasta");
    let reader = IndexedMmapFastaReader::from_fasta(fasta_filename.to_str().unwrap()).unwrap();

    // Note these are the same tests we use in python, but having them here can prevent us from building a wheel with broken code.
    assert_eq!(reader.read_sequence_mmap("chr1:1-1").unwrap(), "A");
    assert_eq!(reader.read_sequence_mmap("chr1:1-2").unwrap(), "AC");
    assert_eq!(reader.read_sequence_mmap("chr1:1-100000").unwrap(), "ACTGACTGACTG");
    assert_eq!(reader.read_sequence_mmap("chr2:1-2").unwrap(), "GG");
    assert_eq!(reader.read_sequence_mmap("chr2:1-1000000").unwrap(), "GGTCAAGGTCAA");
    //Recall to get python based assert_eq!(readering we add 1 to both start and end, so 1-13 is a 12 character string(full sequence)
    assert_eq!(reader.read_sequence_mmap("chr2:1-11").unwrap(), "GGTCAAGGTCA");
    assert_eq!(reader.read_sequence_mmap("chr2:1-12").unwrap(), "GGTCAAGGTCAA");
    assert_eq!(reader.read_sequence_mmap("chr2:1-13").unwrap(), "GGTCAAGGTCAA");

    assert_eq!(reader.read_sequence_mmap("chr3:1-2").unwrap(), "AG");
    assert_eq!(reader.read_sequence_mmap("chr3:1-13").unwrap(), "AGTCAAGGTCCAC");
    assert_eq!(reader.read_sequence_mmap("chr3:1-14").unwrap(), "AGTCAAGGTCCACG"); // adds first character from next line
    assert_eq!(
        reader.read_sequence_mmap("chr3:1-83").unwrap(),
        "AGTCAAGGTCCACGTCAAGGTCCCGGTCAAGGTCCGTGTCAAGGTCCTAGTCAAGGTCAACGTCAAGGTCACGGTCAAGGTCA"
    );
    assert_eq!(
        reader.read_sequence_mmap("chr3:1-84").unwrap(),
        "AGTCAAGGTCCACGTCAAGGTCCCGGTCAAGGTCCGTGTCAAGGTCCTAGTCAAGGTCAACGTCAAGGTCACGGTCAAGGTCAG"
    );
    assert_eq!(
        reader.read_sequence_mmap("chr3:1-10000").unwrap(),
        "AGTCAAGGTCCACGTCAAGGTCCCGGTCAAGGTCCGTGTCAAGGTCCTAGTCAAGGTCAACGTCAAGGTCACGGTCAAGGTCAG"
    );
    assert_eq!(reader.read_sequence_mmap("chr3:84-84").unwrap(), "G");

    // Handles End of assert_eq!(reader
    // Full sequence
    assert_eq!(reader.read_sequence_mmap("chr5:1-1000000").unwrap(), "A");
    // Only one char, should succeed
    assert_eq!(reader.read_sequence_mmap("chr5:1-2").unwrap(), "A");

    // Handles end of multi line but non-full sequence entry
    // Full sequence
    assert_eq!(reader.read_sequence_mmap("chr4:1-16").unwrap(), "CCCCCCCCCCCCACGT");
    assert_eq!(reader.read_sequence_mmap("chr4:1-17").unwrap(), "CCCCCCCCCCCCACGT");
    assert_eq!(
        reader.read_sequence_mmap("chr4:1-1000000").unwrap(),
        "CCCCCCCCCCCCACGT"
    );

    assert_eq!(reader.read_sequence_mmap("chr4:1-17").unwrap(), "CCCCCCCCCCCCACGT");

    assert_eq!(reader.read_sequence_mmap("chr4:3-16").unwrap(), "CCCCCCCCCCACGT");
    assert_eq!(reader.read_sequence_mmap("chr4:17-17").unwrap(), "");
}
