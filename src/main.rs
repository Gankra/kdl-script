use std::{fs::File, io::Read, path::PathBuf, sync::Arc};

use clap::Parser;
use kdl::KdlDocument;
use miette::Diagnostic;
use thiserror::Error;

use parse::KdlScriptParseError;

mod eval;
mod parse;

#[derive(Parser, Debug)]
pub struct Cli {
    pub src: PathBuf,
}

#[derive(Debug, Error, Diagnostic)]
pub enum KdlScriptError {
    #[error(transparent)]
    Io(#[from] std::io::Error),

    #[error(transparent)]
    #[diagnostic(transparent)]
    Kdl(#[from] kdl::KdlError),

    #[error(transparent)]
    #[diagnostic(transparent)]
    Parse(#[from] KdlScriptParseError),
}

pub type Result<T> = std::result::Result<T, KdlScriptError>;

fn main() -> std::result::Result<(), miette::Report> {
    real_main()?;
    Ok(())
}

fn real_main() -> Result<()> {
    let cli = Cli::parse();

    tracing_subscriber::fmt()
        .with_max_level(tracing::level_filters::LevelFilter::WARN)
        .init();

    let input_name = cli.src.display().to_string();
    let mut input_file = File::open(&cli.src)?;
    let mut input_string = String::new();
    input_file.read_to_string(&mut input_string)?;
    let input_string = Arc::new(input_string);

    let src = Arc::new(miette::NamedSource::new(input_name, input_string.clone()));

    let kdl_doc: KdlDocument = input_string.parse::<kdl::KdlDocument>()?;
    let program = parse::parse_kdl_script(&src, kdl_doc)?;
    let result = eval::eval_kdl_script(&src, program)?;

    println!("{}", result);
    Ok(())
}
