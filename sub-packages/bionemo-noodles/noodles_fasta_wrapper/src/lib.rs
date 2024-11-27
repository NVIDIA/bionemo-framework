use std::fs::File;
use pyo3::prelude::*;

use noodles_fasta::{self as fasta, fai};
use noodles_fasta::fai::Record;
use noodles_core::Region;
use std::path::{Path};

// Expose the Record struct so we can package it nicely in Python.
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

#[pyclass]
struct IndexedFastaReader {
    reader: fasta::io::IndexedReader<fasta::io::BufReader<File>>,
}

#[pymethods]
impl IndexedFastaReader {
    #[new]
    fn new(fasta_path: &str) -> PyResult<Self> {
        let fai_path = fasta_path.to_string() + ".fai";
        let fai_path = Path::new(&fai_path);  // Convert back to a Path


        let fasta_path = Path::new(fasta_path);
        // let fai_path = fasta_path.with_extension("fai");

        // Check if the .fai index file exists; if not, create it.
        if !fai_path.exists() {
            // Generate the index by reading the FASTA file
            let index = fasta::io::index(fasta_path)
                .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("Failed to create index: {}", e)))?;

            // Write the index to the .fai file
            let fai_file = File::create(&fai_path)
                .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("Failed to create .fai file: {}", e)))?;

            let mut writer = fai::Writer::new(fai_file);
            writer.write_index(&index)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Failed to write .fai index: {}", e)))?;
        }


        let reader = fasta::io::indexed_reader::Builder::default()
            .build_from_path(fasta_path)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("Failed to create indexed reader: {}", e)))?;
        
        Ok(IndexedFastaReader { reader })
    }

    fn query_region(&mut self, region_str: &str) -> PyResult<String> {
        let region = region_str.parse()
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Invalid region: {}", e)))?;
        

        let query_result = self.reader.query(&region)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Failed to query region: {}", e)))?;

        
        Ok(
            String::from_utf8_lossy(
                query_result.sequence().as_ref()
            ).to_string()
        )
    }

    fn records(&self) -> Vec<PyRecord> {
        // get all the entries in the index, useful for building bounds in python land.
        self.reader
            .index()
            .as_ref()
            .iter()
            .map(|record| PyRecord::from(record))
            .collect()
    }
    
}

#[pymodule]
fn noodles_fasta_wrapper(_: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<IndexedFastaReader>()?;
    Ok(())
}

