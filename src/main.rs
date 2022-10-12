use std::path::PathBuf;

use clap::Parser;
use kdl_script::Compiler;

#[derive(Parser, Debug)]
pub struct Cli {
    pub src: PathBuf,
}

fn main() -> std::result::Result<(), miette::Report> {
    real_main()?;
    Ok(())
}

fn real_main() -> std::result::Result<(), miette::Report> {
    let cli = Cli::parse();

    tracing_subscriber::fmt()
        .with_max_level(tracing::level_filters::LevelFilter::WARN)
        .init();

    let mut compiler = Compiler::new();
    let typed = compiler.compile_path(&cli.src)?;
    println!("{:?}", typed);

    let result = compiler.eval()?;
    if let Some(result) = result {
        println!("{}", result);
    }
    Ok(())
}

/*
fn backend_to_the_future(program: &Arc<TypedProgram>) {

}

fn emit_types_for_funcs
*/
