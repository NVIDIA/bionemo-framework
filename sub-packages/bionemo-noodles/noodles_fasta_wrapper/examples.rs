//! This writes the output to stdout rather than `<src>.fai`.
//!
//! The result matches the output of `samtools faidx <src>`.

// NOTE we probably want this to be a function and not a CLI tool.

use std::{env, io};

use noodles_fasta::{self as fasta, fai};

fn main() -> Result<(), Box<dyn std::error::Error>> {

    let mut args = env::args().skip(1);
    // parse the arg that gets the path
    let src: String = args.next().expect("missing src");

    // userful for our query downstream
    let raw_region: String = args.next().expect("missing region");

    let mut reader = fasta::io::indexed_reader::Builder::default().build_from_path(src)?;

    // noodles_core::region::Region is the type here.
    let region = raw_region.parse()?;
    let record = reader.query(&region)?;


    Ok(())
}
