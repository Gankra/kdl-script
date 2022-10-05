use std::sync::Arc;

use kdl::{KdlDocument, KdlEntry, KdlNode};
use miette::{Diagnostic, NamedSource, SourceSpan};
use thiserror::Error;
use tracing::trace;

use crate::Result;

pub type Ident = String;
pub type TyName = String;

#[derive(Debug, Error, Diagnostic)]
#[error("{message}")]
pub struct KdlScriptParseError {
    pub message: String,
    #[source_code]
    pub src: Arc<NamedSource>,
    #[label]
    pub span: SourceSpan,
    #[help]
    pub help: Option<String>,
}

#[derive(Debug, Clone)]
pub struct KdlScriptProgram {
    pub tys: Vec<TyDecl>,
    pub funcs: Vec<FuncDecl>,
}

#[derive(Debug, Clone)]
pub enum Attr {
    Derive(AttrDerive),
    Packed(AttrPacked),
}

#[derive(Debug, Clone)]
pub struct AttrPacked {}

#[derive(Debug, Clone)]
pub struct AttrDerive(Vec<Spanned<String>>);

#[derive(Debug, Clone)]
pub enum TyDecl {
    Struct(StructDecl),
}

#[derive(Debug, Clone)]
pub struct StructDecl {
    pub name: Spanned<Ident>,
    pub fields: Vec<TypedVar>,
    pub attrs: Vec<Attr>,
}

#[derive(Debug, Clone)]
pub struct TypedVar {
    pub name: Option<Spanned<Ident>>,
    pub ty: Spanned<TyName>,
}

#[derive(Debug, Clone)]
pub struct FuncDecl {
    pub name: Spanned<Ident>,
    pub inputs: Vec<TypedVar>,
    pub outputs: Vec<TypedVar>,
    pub attrs: Vec<Attr>,
    pub body: Vec<Stmt>,
}

#[derive(Debug, Clone)]
pub enum Stmt {
    Let(LetStmt),
    Return(ReturnStmt),
    Print(PrintStmt),
}

#[derive(Debug, Clone)]
pub struct LetStmt {
    pub var: Option<Spanned<Ident>>,
    pub expr: Spanned<Expr>,
}

#[derive(Debug, Clone)]
pub struct ReturnStmt {
    pub expr: Spanned<Expr>,
}

#[derive(Debug, Clone)]
pub struct PrintStmt {
    pub expr: Spanned<Expr>,
}

#[derive(Debug, Clone)]
pub enum Expr {
    Call(CallExpr),
    Path(PathExpr),
    Ctor(CtorExpr),
    Literal(LiteralExpr),
}

#[derive(Debug, Clone)]
pub struct CallExpr {
    pub func: Spanned<Ident>,
    pub args: Vec<Spanned<Expr>>,
}

#[derive(Debug, Clone)]
pub struct PathExpr {
    pub var: Spanned<Ident>,
    pub path: Vec<Spanned<Ident>>,
}

#[derive(Debug, Clone)]
pub struct CtorExpr {
    pub ty: Spanned<Ident>,
    pub vals: Vec<Spanned<LetStmt>>,
}

#[derive(Debug, Clone)]
pub struct LiteralExpr {
    pub span: SourceSpan,
    pub val: Literal,
}

#[derive(Debug, Clone)]
pub enum Literal {
    Float(f64),
    Int(i64),
    Bool(bool),
}

pub fn parse_kdl_script(src: &Arc<NamedSource>, ast: KdlDocument) -> Result<KdlScriptProgram> {
    trace!("parsing");
    let mut tys = vec![];
    let mut funcs = vec![];

    let mut cur_attrs = vec![];
    for node in ast.nodes() {
        let name = node.name().value();
        let entries = node.entries();
        // let mut entries_iter = entries.into_iter();
        match name {
            "fn" => {
                trace!("fn");
                let name = one_string(src, node, "function name")?;
                let attrs = std::mem::replace(&mut cur_attrs, vec![]);
                let mut inputs = vec![];
                let mut outputs = vec![];
                let mut body = vec![];

                let mut reached_body = false;
                let mut input_span = None;
                let mut output_span = None;
                for stmt in node.children().into_iter().flat_map(|d| d.nodes()) {
                    match stmt.name().value() {
                        "inputs" => {
                            trace!("fn input");
                            if reached_body {
                                return Err(KdlScriptParseError {
                                    message: format!("input declaration must come before the body"),
                                    src: src.clone(),
                                    span: *stmt.name().span(),
                                    help: None,
                                })?;
                            }
                            if let Some(_old_input) = input_span {
                                return Err(KdlScriptParseError {
                                    message: format!("duplicate input block"),
                                    src: src.clone(),
                                    span: *stmt.name().span(),
                                    help: None,
                                })?;
                            }
                            if let Some(_old_output) = output_span {
                                return Err(KdlScriptParseError {
                                    message: format!(
                                        "It's confusing to declare inputs after outputs"
                                    ),
                                    src: src.clone(),
                                    span: *stmt.name().span(),
                                    help: Some("Move this before the output block".to_string()),
                                })?;
                            }
                            no_args(src, stmt)?;
                            inputs = typed_var_children(src, stmt)?;
                            input_span = Some(*stmt.name().span());
                            continue;
                        }
                        "outputs" => {
                            trace!("fn output");
                            if reached_body {
                                return Err(KdlScriptParseError {
                                    message: format!(
                                        "output declaration must come before the body"
                                    ),
                                    src: src.clone(),
                                    span: *stmt.name().span(),
                                    help: None,
                                })?;
                            }
                            if let Some(_old_output) = output_span {
                                return Err(KdlScriptParseError {
                                    message: format!("duplicate output block"),
                                    src: src.clone(),
                                    span: *stmt.name().span(),
                                    help: None,
                                })?;
                            }
                            no_args(src, stmt)?;
                            outputs = typed_var_children(src, stmt)?;
                            output_span = Some(*stmt.name().span());
                            continue;
                        }
                        "let" => {
                            trace!("let stmt");
                            let name = string_at(src, stmt, "variable name", 0)?;
                            let name = if &*name == "_" { None } else { Some(name) };
                            let expr = expr_rhs(src, stmt, 1)?;
                            body.push(Stmt::Let(LetStmt { var: name, expr }));
                        }
                        "return" => {
                            trace!("return stmt");
                            let expr = expr_rhs(src, stmt, 0)?;
                            body.push(Stmt::Return(ReturnStmt { expr }));
                        }
                        "print" => {
                            trace!("print stmt");
                            let expr = expr_rhs(src, stmt, 0)?;
                            body.push(Stmt::Print(PrintStmt { expr }));
                        }
                        x => {
                            return Err(KdlScriptParseError {
                                message: format!("I don't know what a '{x}' statement is"),
                                src: src.clone(),
                                span: *stmt.name().span(),
                                help: None,
                            })?;
                        }
                    }
                    reached_body = true;
                }

                funcs.push(FuncDecl {
                    name,
                    inputs,
                    outputs,
                    body,
                    attrs,
                });
            }
            "struct" => {
                trace!("struct decl");
                let name = one_string(src, node, "type name")?;
                let attrs = std::mem::replace(&mut cur_attrs, vec![]);
                let fields = typed_var_children(src, node)?;

                tys.push(TyDecl::Struct(StructDecl {
                    name,
                    fields,
                    attrs,
                }))
            }
            "@packed" => {
                trace!("packed attr");
                no_children(src, node)?;
                cur_attrs.push(Attr::Packed(AttrPacked {}));
            }
            "@derive" => {
                trace!("derive attr");
                let traits = string_list(src, entries)?;
                no_children(src, node)?;
                cur_attrs.push(Attr::Derive(AttrDerive(traits)));
            }
            x => {
                return Err(KdlScriptParseError {
                    message: format!("I don't know what a '{x}' is"),
                    src: src.clone(),
                    span: *node.name().span(),
                    help: None,
                })?;
            }
        }
    }

    // Add builtins jankily
    funcs.push(FuncDecl {
        name: Spanned::new(String::from("+"), (0..0).into()),
        inputs: vec![
            TypedVar {
                name: Some(Spanned::new(String::from("lhs"), (0..0).into())),
                ty: Spanned::new(String::from("i64"), (0..0).into()),
            },
            TypedVar {
                name: Some(Spanned::new(String::from("rhs"), (0..0).into())),
                ty: Spanned::new(String::from("i64"), (0..0).into()),
            },
        ],
        outputs: vec![TypedVar {
            name: None,
            ty: Spanned::new(String::from("i64"), (0..0).into()),
        }],
        attrs: vec![],
        body: vec![],
    });

    let program = KdlScriptProgram { tys, funcs };
    Ok(program)
}

fn string_list(src: &Arc<NamedSource>, entries: &[KdlEntry]) -> Result<Vec<Spanned<String>>> {
    entries
        .into_iter()
        .map(|e| -> Result<Spanned<String>> {
            if e.name().is_some() {
                return Err(KdlScriptParseError {
                    message: format!("Named values don't belong here, only strings"),
                    src: src.clone(),
                    span: *e.span(),
                    help: Some("try removing the name".to_owned()),
                })?;
            }
            match e.value() {
                kdl::KdlValue::RawString(s) | kdl::KdlValue::String(s) => {
                    Ok(Spanned::new(s.clone(), *e.span()))
                }
                _ => {
                    return Err(KdlScriptParseError {
                        message: format!("This should be a string"),
                        src: src.clone(),
                        span: *e.span(),
                        help: Some("try adding quotes?".to_owned()),
                    })?;
                }
            }
        })
        .collect()
}

fn one_string(src: &Arc<NamedSource>, node: &KdlNode, desc: &str) -> Result<Spanned<String>> {
    let res = string_at(src, node, desc, 0)?;
    let entries = node.entries();
    if let Some(e) = entries.get(1) {
        return Err(KdlScriptParseError {
            message: format!("You have something extra after your {desc}"),
            src: src.clone(),
            span: *e.span(),
            help: Some("remove this?".to_owned()),
        })?;
    }
    Ok(res)
}

fn string_at(
    src: &Arc<NamedSource>,
    node: &KdlNode,
    desc: &str,
    offset: usize,
) -> Result<Spanned<String>> {
    let entries = node.entries();
    if let Some(e) = entries.get(offset) {
        if e.name().is_some() {
            return Err(KdlScriptParseError {
                message: format!("Named values don't belong here, only strings"),
                src: src.clone(),
                span: *e.span(),
                help: Some("try removing the name".to_owned()),
            })?;
        }

        match e.value() {
            kdl::KdlValue::RawString(s) | kdl::KdlValue::String(s) => {
                Ok(Spanned::new(s.clone(), *e.span()))
            }
            _ => {
                return Err(KdlScriptParseError {
                    message: format!("This should be a {desc} (string)"),
                    src: src.clone(),
                    span: *e.span(),
                    help: Some("try adding quotes".to_owned()),
                })?;
            }
        }
    } else {
        let node_ident = node.name().span();
        let after_ident = node_ident.offset() + node_ident.len();
        return Err(KdlScriptParseError {
            message: format!("Hey I need a {desc} (string) here!"),
            src: src.clone(),
            span: (after_ident..after_ident).into(),
            help: None,
        })?;
    }
}

fn no_args(src: &Arc<NamedSource>, node: &KdlNode) -> Result<()> {
    if let Some(entry) = node.entries().get(0) {
        return Err(KdlScriptParseError {
            message: format!("This shouldn't have arguments"),
            src: src.clone(),
            span: *entry.span(),
            help: Some("delete them?".to_string()),
        })?;
    }
    Ok(())
}

fn no_children(src: &Arc<NamedSource>, node: &KdlNode) -> Result<()> {
    if let Some(children) = node.children() {
        return Err(KdlScriptParseError {
            message: format!("These children should never have been born"),
            src: src.clone(),
            span: *children.span(),
            help: Some("delete this block?".to_string()),
        })?;
    }
    Ok(())
}

fn typed_var_children(src: &Arc<NamedSource>, node: &KdlNode) -> Result<Vec<TypedVar>> {
    node.children()
        .into_iter()
        .flat_map(|d| d.nodes())
        .map(|var| {
            let name = var_name_decl(src, var)?;
            let ty = one_string(src, var, "type")?;
            no_children(src, var)?;
            Ok(TypedVar { name, ty })
        })
        .collect()
}

fn var_name_decl(_src: &Arc<NamedSource>, var: &KdlNode) -> Result<Option<Spanned<Ident>>> {
    let name = var.name();
    let name = if name.value() == "_" {
        None
    } else {
        Some(Spanned::new(name.value().to_owned(), *name.span()))
    };
    Ok(name)
}

fn expr_rhs(src: &Arc<NamedSource>, node: &KdlNode, expr_start: usize) -> Result<Spanned<Expr>> {
    trace!("expr rhs");
    let expr = if let Ok(string) = string_at(src, node, "", expr_start) {
        if let Some((func, "")) = string.rsplit_once(':') {
            trace!("  call expr");
            let func = Spanned::new(func.to_owned(), *Spanned::span(&string));
            let args = func_args(src, node, expr_start + 1)?;
            Expr::Call(CallExpr { func, args })
        } else if node.children().is_some() {
            trace!("  ctor expr");
            let ty = string;
            let vals = let_stmt_children(src, node)?;
            Expr::Ctor(CtorExpr { ty, vals })
        } else {
            trace!("  path expr");
            let mut parts = string.split('.');
            let var = Spanned::new(parts.next().unwrap().to_owned(), *Spanned::span(&string));
            let path = parts
                .map(|s| Spanned::new(s.to_owned(), *Spanned::span(&string)))
                .collect();
            Expr::Path(PathExpr { var, path })
        }
    } else if let Some(val) = node.entries().get(expr_start) {
        trace!("  literal expr");
        Expr::Literal(literal_expr(src, val)?)
    } else {
        return Err(KdlScriptParseError {
            message: format!("I thought there was supposed to be an expression after here?"),
            src: src.clone(),
            span: *node.span(),
            help: None,
        })?;
    };

    Ok(Spanned::new(expr, *node.span()))
}

fn smol_expr(src: &Arc<NamedSource>, node: &KdlNode, expr_at: usize) -> Result<Spanned<Expr>> {
    trace!("smol expr");
    let expr = if let Ok(string) = string_at(src, node, "", expr_at) {
        if let Some((_func, "")) = string.rsplit_once(':') {
            return Err(KdlScriptParseError {
                message: format!(
                    "Nested function calls aren't supported because this is a shitpost"
                ),
                src: src.clone(),
                span: *node.span(),
                help: None,
            })?;
        } else if node.children().is_some() {
            return Err(KdlScriptParseError {
                message: format!(
                    "Ctors exprs can't be nested in function calls because this is a shitpost"
                ),
                src: src.clone(),
                span: *node.span(),
                help: None,
            })?;
        } else {
            trace!("  path expr");
            let mut parts = string.split('.');
            let var = Spanned::new(parts.next().unwrap().to_owned(), *Spanned::span(&string));
            let path = parts
                .map(|s| Spanned::new(s.to_owned(), *Spanned::span(&string)))
                .collect();
            Expr::Path(PathExpr { var, path })
        }
    } else if let Some(val) = node.entries().get(expr_at) {
        trace!("  literal expr");
        Expr::Literal(literal_expr(src, val)?)
    } else {
        return Err(KdlScriptParseError {
            message: format!("I thought there was supposed to be an expression after here?"),
            src: src.clone(),
            span: *node.span(),
            help: None,
        })?;
    };

    Ok(Spanned::new(expr, *node.span()))
}

fn func_args(
    src: &Arc<NamedSource>,
    node: &KdlNode,
    expr_start: usize,
) -> Result<Vec<Spanned<Expr>>> {
    node.entries()[expr_start..]
        .iter()
        .enumerate()
        .map(|(idx, _e)| smol_expr(src, node, expr_start + idx))
        .collect()
}

fn let_stmt_children(src: &Arc<NamedSource>, node: &KdlNode) -> Result<Vec<Spanned<LetStmt>>> {
    node.children()
        .into_iter()
        .flat_map(|d| d.nodes())
        .map(|var| {
            let name = var_name_decl(src, var)?;
            let expr = expr_rhs(src, var, 0)?;
            Ok(Spanned::new(LetStmt { var: name, expr }, *var.span()))
        })
        .collect()
}

fn literal_expr(src: &Arc<NamedSource>, entry: &KdlEntry) -> Result<LiteralExpr> {
    if entry.name().is_some() {
        return Err(KdlScriptParseError {
            message: format!("Named values don't belong here, only literals"),
            src: src.clone(),
            span: *entry.span(),
            help: Some("try removing the name".to_owned()),
        })?;
    }

    let val = match entry.value() {
        kdl::KdlValue::RawString(_) | kdl::KdlValue::String(_) => {
            return Err(KdlScriptParseError {
                message: format!("strings aren't supported literals"),
                src: src.clone(),
                span: *entry.span(),
                help: None,
            })?;
        }
        kdl::KdlValue::Null => {
            return Err(KdlScriptParseError {
                message: format!("nulls aren't supported literals"),
                src: src.clone(),
                span: *entry.span(),
                help: None,
            })?;
        }
        kdl::KdlValue::Base2(int)
        | kdl::KdlValue::Base8(int)
        | kdl::KdlValue::Base10(int)
        | kdl::KdlValue::Base16(int) => Literal::Int(*int),
        kdl::KdlValue::Base10Float(val) => Literal::Float(*val),
        kdl::KdlValue::Bool(val) => Literal::Bool(*val),
    };

    Ok(LiteralExpr {
        span: *entry.span(),
        val,
    })
}

#[derive(Debug, Clone)]
pub struct Spanned<T> {
    span: SourceSpan,
    val: T,
}
impl<T> Spanned<T> {
    pub fn new(val: T, span: SourceSpan) -> Self {
        Self { val, span }
    }

    pub fn span(this: &Self) -> &SourceSpan {
        &this.span
    }
}

impl<T> std::ops::Deref for Spanned<T> {
    type Target = T;
    fn deref(&self) -> &Self::Target {
        &self.val
    }
}
