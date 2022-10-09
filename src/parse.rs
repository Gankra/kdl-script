use std::sync::Arc;

use kdl::{KdlDocument, KdlEntry, KdlNode};
use miette::{Diagnostic, NamedSource, SourceSpan};
use thiserror::Error;
use tracing::trace;

use crate::spanned::Spanned;
use crate::{Compiler, Result};

pub type Ident = Spanned<String>;
pub type StableMap<K, V> = linked_hash_map::LinkedHashMap<K, V>;

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
pub struct ParsedProgram {
    pub tys: StableMap<Ident, TyDecl>,
    pub funcs: StableMap<Ident, FuncDecl>,
}

#[derive(Debug, Clone)]
pub enum TyDecl {
    Struct(StructDecl),
    Union(UnionDecl),
    Enum(EnumDecl),
    Tagged(TaggedDecl),
    Alias(AliasDecl),
    Pun(PunDecl),
}

#[derive(Debug, Clone)]
pub enum TyRef {
    Name(Ident),
    Array(Box<TyRef>, u64),
    Ref(Box<TyRef>),
    Empty,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Attr {
    Derive(AttrDerive),
    Packed(AttrPacked),
    Passthrough(AttrPassthrough),
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct AttrPacked {}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct AttrDerive(Vec<Spanned<String>>);

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct AttrPassthrough(Spanned<String>);

#[derive(Debug, Clone)]
pub struct StructDecl {
    pub name: Ident,
    pub fields: Vec<TypedVar>,
    pub attrs: Vec<Attr>,
}

#[derive(Debug, Clone)]
pub struct UnionDecl {
    pub name: Ident,
    pub fields: Vec<TypedVar>,
    pub attrs: Vec<Attr>,
}

#[derive(Debug, Clone)]
pub struct EnumDecl {
    pub name: Ident,
    pub variants: Vec<EnumVariant>,
    pub attrs: Vec<Attr>,
}

#[derive(Debug, Clone)]
pub struct EnumVariant {
    pub name: Ident,
    pub val: Option<LiteralExpr>,
}

#[derive(Debug, Clone)]
pub struct TaggedDecl {
    pub name: Ident,
    pub variants: Vec<TaggedVariant>,
    pub attrs: Vec<Attr>,
}

#[derive(Debug, Clone)]
pub struct TaggedVariant {
    pub name: Ident,
    pub fields: Option<Vec<TypedVar>>,
}

#[derive(Debug, Clone)]
pub struct PunDecl {
    pub name: Ident,
    pub blocks: Vec<PunBlock>,
    pub attrs: Vec<Attr>,
}

#[derive(Debug, Clone)]
pub struct PunBlock {
    pub selector: PunSelector,
    pub decl: TyDecl,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum PunSelector {
    Any(Vec<PunSelector>),
    // All(Vec<PunSelector>),
    Lang(Spanned<String>),
    Default,
}

#[derive(Debug, Clone)]
pub struct PunEnv {
    pub lang: String,
    // compiler: String,
    // os: String,
    // cpu: String,
}

impl PunSelector {
    fn matches(&self, env: &PunEnv) -> bool {
        use PunSelector::*;
        match self {
            Any(args) => args.iter().any(|s| s.matches(env)),
            // All(args) => args.iter().all(|s| s.matches(env)),
            Lang(lang) => &env.lang == &**lang,
            Default => true,
        }
    }
}

#[derive(Debug, Clone)]
pub struct AliasDecl {
    pub name: Ident,
    pub alias: TyRef,
    pub attrs: Vec<Attr>,
}

#[derive(Debug, Clone)]
pub struct TypedVar {
    pub name: Option<Ident>,
    pub ty: TyRef,
}

#[derive(Debug, Clone)]
pub struct FuncDecl {
    pub name: Ident,
    pub inputs: Vec<TypedVar>,
    pub outputs: Vec<TypedVar>,
    pub attrs: Vec<Attr>,
    pub body: Vec<Stmt>,
}

struct Parser<'a> {
    comp: &'a mut Compiler,
    src: Arc<NamedSource>,
    ast: &'a KdlDocument,
}

pub fn parse_kdl_script(
    comp: &mut Compiler,
    src: Arc<NamedSource>,
    ast: &KdlDocument,
) -> Result<ParsedProgram> {
    let mut parser = Parser { comp, src, ast };
    parser.parse()
}

impl Parser<'_> {
    fn parse(&mut self) -> Result<ParsedProgram> {
        trace!("parsing");

        let mut program = self.parse_module(self.ast)?;
        program.add_builtins()?;

        Ok(program)
    }

    fn parse_module(&mut self, doc: &KdlDocument) -> Result<ParsedProgram> {
        let mut funcs = StableMap::new();
        let mut tys = StableMap::new();

        let mut cur_attrs = vec![];
        for node in doc.nodes() {
            let name = node.name().value();

            // If it's an attribute, gather it up to be given to a "real" item
            if name.starts_with('@') {
                cur_attrs.push(self.attr(node)?);
                continue;
            }

            // Ok it's a real item, grab all the attributes, they belong to it
            let attrs = std::mem::replace(&mut cur_attrs, vec![]);

            match name {
                "fn" => {
                    let func = self.func_decl(node, attrs)?;
                    funcs.insert(func.name.clone(), func);
                }
                "struct" => {
                    let ty = self.struct_decl(node, attrs)?;
                    let old = tys.insert(ty.name.clone(), TyDecl::Struct(ty));
                    assert!(old.is_none(), "duplicate type def");
                }
                "union" => {
                    let ty = self.union_decl(node, attrs)?;
                    let old = tys.insert(ty.name.clone(), TyDecl::Union(ty));
                    assert!(old.is_none(), "duplicate type def");
                }
                "enum" => {
                    let ty = self.enum_decl(node, attrs)?;
                    let old = tys.insert(ty.name.clone(), TyDecl::Enum(ty));
                    assert!(old.is_none(), "duplicate type def");
                }
                "tagged" => {
                    let ty = self.tagged_decl(node, attrs)?;
                    let old = tys.insert(ty.name.clone(), TyDecl::Tagged(ty));
                    assert!(old.is_none(), "duplicate type def");
                }
                "alias" => {
                    let ty = self.alias_decl(node, attrs)?;
                    let old = tys.insert(ty.name.clone(), TyDecl::Alias(ty));
                    assert!(old.is_none(), "duplicate type def");
                }
                "pun" => {
                    let ty = self.pun_decl(node, attrs)?;
                    let old = tys.insert(ty.name.clone(), TyDecl::Pun(ty));
                    assert!(old.is_none(), "duplicate type def");
                }

                // "union" =>
                // "enum" =>
                // "tagged" =>
                x => {
                    return Err(KdlScriptParseError {
                        message: format!("I don't know what a '{x}' is"),
                        src: self.src.clone(),
                        span: *node.name().span(),
                        help: None,
                    })?;
                }
            }
        }

        Ok(ParsedProgram { tys, funcs })
    }

    fn struct_decl(&mut self, node: &KdlNode, attrs: Vec<Attr>) -> Result<StructDecl> {
        trace!("struct decl");
        let name = self.one_string(node, "type name")?;
        let fields = self.typed_var_children(node)?;

        Ok(StructDecl {
            name,
            fields,
            attrs,
        })
    }

    fn union_decl(&mut self, node: &KdlNode, attrs: Vec<Attr>) -> Result<UnionDecl> {
        trace!("union decl");
        let name = self.one_string(node, "type name")?;
        let fields = self.typed_var_children(node)?;

        Ok(UnionDecl {
            name,
            fields,
            attrs,
        })
    }

    fn enum_decl(&mut self, node: &KdlNode, attrs: Vec<Attr>) -> Result<EnumDecl> {
        trace!("enum decl");
        let name = self.one_string(node, "type name")?;
        let variants = self.enum_variant_children(node)?;

        Ok(EnumDecl {
            name,
            variants,
            attrs,
        })
    }

    fn tagged_decl(&mut self, node: &KdlNode, attrs: Vec<Attr>) -> Result<TaggedDecl> {
        trace!("enum decl");
        let name = self.one_string(node, "type name")?;
        let variants = self.tagged_variant_children(node)?;

        Ok(TaggedDecl {
            name,
            variants,
            attrs,
        })
    }

    fn pun_decl(&mut self, node: &KdlNode, attrs: Vec<Attr>) -> Result<PunDecl> {
        let name = self.one_string(node, "type name")?;

        let mut blocks = vec![];
        for item in node.children().into_iter().flat_map(|d| d.nodes()) {
            let item_name = item.name().value();
            match item_name {
                "lang" => {
                    let langs = self.string_list(item.entries())?;
                    let final_ty = self.pun_block(item, &name)?;
                    blocks.push(PunBlock {
                        selector: PunSelector::Any(
                            langs.into_iter().map(PunSelector::Lang).collect(),
                        ),
                        decl: final_ty,
                    });
                }
                "default" => {
                    self.no_args(item)?;
                    let final_ty = self.pun_block(item, &name)?;
                    blocks.push(PunBlock {
                        selector: PunSelector::Default,
                        decl: final_ty,
                    });
                }
                x => {
                    return Err(KdlScriptParseError {
                        message: format!("I don't know what a '{x}' is"),
                        src: self.src.clone(),
                        span: *item.name().span(),
                        help: None,
                    })?;
                }
            }
        }

        Ok(PunDecl {
            name,
            blocks,
            attrs,
        })
    }

    fn pun_block(&mut self, block: &KdlNode, final_ty_name: &Ident) -> Result<TyDecl> {
        if let Some(doc) = block.children() {
            // Recursively parse this block as an entire KdlScript program
            let defs = self.parse_module(doc)?;

            // Don't want any functions
            if let Some((_name, func)) = defs.funcs.iter().next() {
                return Err(KdlScriptParseError {
                    message: format!("puns can't contain function decls"),
                    src: self.src.clone(),
                    span: Spanned::span(&func.name),
                    help: None,
                })?;
            }

            let mut final_ty = None;

            // Only want one type declared (might loosen this later)
            for (ty_name, ty) in defs.tys {
                if &ty_name == final_ty_name {
                    // this is the type
                    final_ty = Some(ty);
                } else {
                    return Err(KdlScriptParseError {
                        message: format!("pun declared a type other than what it should have"),
                        src: self.src.clone(),
                        span: Spanned::span(&ty_name),
                        help: None,
                    })?;
                }
            }

            // Check that we defined the type
            if let Some(ty) = final_ty {
                Ok(ty)
            } else {
                return Err(KdlScriptParseError {
                    message: format!("pun block failed to define the type it puns!"),
                    src: self.src.clone(),
                    span: *block.span(),
                    help: None,
                })?;
            }
        } else {
            return Err(KdlScriptParseError {
                message: format!("lang blocks need bodys"),
                src: self.src.clone(),
                span: *block.span(),
                help: None,
            })?;
        }
    }

    fn alias_decl(&mut self, node: &KdlNode, attrs: Vec<Attr>) -> Result<AliasDecl> {
        let name = self.string_at(node, "type name", 0)?;
        let alias_str = self.string_at(node, "type name", 1)?;
        let alias = self.ty_ref(&alias_str)?;

        Ok(AliasDecl { name, alias, attrs })
    }

    fn func_decl(&mut self, node: &KdlNode, attrs: Vec<Attr>) -> Result<FuncDecl> {
        trace!("fn");
        let name = self.one_string(node, "function name")?;
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
                            src: self.src.clone(),
                            span: *stmt.name().span(),
                            help: None,
                        })?;
                    }
                    if let Some(_old_input) = input_span {
                        return Err(KdlScriptParseError {
                            message: format!("duplicate input block"),
                            src: self.src.clone(),
                            span: *stmt.name().span(),
                            help: None,
                        })?;
                    }
                    if let Some(_old_output) = output_span {
                        return Err(KdlScriptParseError {
                            message: format!("It's confusing to declare inputs after outputs"),
                            src: self.src.clone(),
                            span: *stmt.name().span(),
                            help: Some("Move this before the output block".to_string()),
                        })?;
                    }
                    self.no_args(stmt)?;
                    inputs = self.typed_var_children(stmt)?;
                    input_span = Some(*stmt.name().span());
                    continue;
                }
                "outputs" => {
                    trace!("fn output");
                    if reached_body {
                        return Err(KdlScriptParseError {
                            message: format!("output declaration must come before the body"),
                            src: self.src.clone(),
                            span: *stmt.name().span(),
                            help: None,
                        })?;
                    }
                    if let Some(_old_output) = output_span {
                        return Err(KdlScriptParseError {
                            message: format!("duplicate output block"),
                            src: self.src.clone(),
                            span: *stmt.name().span(),
                            help: None,
                        })?;
                    }
                    self.no_args(stmt)?;
                    outputs = self.typed_var_children(stmt)?;
                    output_span = Some(*stmt.name().span());
                    continue;
                }
                "let" => {
                    trace!("let stmt");
                    let name = self.string_at(stmt, "variable name", 0)?;
                    let name = if &*name == "_" { None } else { Some(name) };
                    let expr = self.expr_rhs(stmt, 1)?;
                    body.push(Stmt::Let(LetStmt { var: name, expr }));
                }
                "return" => {
                    trace!("return stmt");
                    let expr = self.expr_rhs(stmt, 0)?;
                    body.push(Stmt::Return(ReturnStmt { expr }));
                }
                "print" => {
                    trace!("print stmt");
                    let expr = self.expr_rhs(stmt, 0)?;
                    body.push(Stmt::Print(PrintStmt { expr }));
                }
                x => {
                    return Err(KdlScriptParseError {
                        message: format!("I don't know what a '{x}' statement is"),
                        src: self.src.clone(),
                        span: *stmt.name().span(),
                        help: None,
                    })?;
                }
            }
            reached_body = true;
        }

        Ok(FuncDecl {
            name,
            inputs,
            outputs,
            body,
            attrs,
        })
    }

    fn attr(&mut self, attr: &KdlNode) -> Result<Attr> {
        let entries = attr.entries();
        let attr = match attr.name().value() {
            "@packed" => {
                trace!("packed attr");
                self.no_children(attr)?;
                Attr::Packed(AttrPacked {})
            }
            "@derive" => {
                trace!("derive attr");
                let traits = self.string_list(entries)?;
                self.no_children(attr)?;
                Attr::Derive(AttrDerive(traits))
            }
            "@" => {
                trace!("passthrough attr");
                let val = self.one_string(attr, "attribute to pass through to target language")?;
                Attr::Passthrough(AttrPassthrough(val))
            }
            x => {
                return Err(KdlScriptParseError {
                    message: format!("I don't know what a '{x}' attribute is"),
                    src: self.src.clone(),
                    span: *attr.name().span(),
                    help: None,
                })?;
            }
        };
        Ok(attr)
    }

    fn string_list(&mut self, entries: &[KdlEntry]) -> Result<Vec<Spanned<String>>> {
        entries
            .into_iter()
            .map(|e| -> Result<Spanned<String>> {
                if e.name().is_some() {
                    return Err(KdlScriptParseError {
                        message: format!("Named values don't belong here, only strings"),
                        src: self.src.clone(),
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
                            src: self.src.clone(),
                            span: *e.span(),
                            help: Some("try adding quotes?".to_owned()),
                        })?;
                    }
                }
            })
            .collect()
    }

    fn one_string(&mut self, node: &KdlNode, desc: &str) -> Result<Spanned<String>> {
        let res = self.string_at(node, desc, 0)?;
        let entries = node.entries();
        if let Some(e) = entries.get(1) {
            return Err(KdlScriptParseError {
                message: format!("You have something extra after your {desc}"),
                src: self.src.clone(),
                span: *e.span(),
                help: Some("remove this?".to_owned()),
            })?;
        }
        Ok(res)
    }

    fn string_at(&mut self, node: &KdlNode, desc: &str, offset: usize) -> Result<Spanned<String>> {
        let entries = node.entries();
        if let Some(e) = entries.get(offset) {
            if e.name().is_some() {
                return Err(KdlScriptParseError {
                    message: format!("Named values don't belong here, only strings"),
                    src: self.src.clone(),
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
                        src: self.src.clone(),
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
                src: self.src.clone(),
                span: (after_ident..after_ident).into(),
                help: None,
            })?;
        }
    }

    fn no_args(&mut self, node: &KdlNode) -> Result<()> {
        if let Some(entry) = node.entries().get(0) {
            return Err(KdlScriptParseError {
                message: format!("This shouldn't have arguments"),
                src: self.src.clone(),
                span: *entry.span(),
                help: Some("delete them?".to_string()),
            })?;
        }
        Ok(())
    }

    fn no_children(&mut self, node: &KdlNode) -> Result<()> {
        if let Some(children) = node.children() {
            return Err(KdlScriptParseError {
                message: format!("These children should never have been born"),
                src: self.src.clone(),
                span: *children.span(),
                help: Some("delete this block?".to_string()),
            })?;
        }
        Ok(())
    }

    fn typed_var_children(&mut self, node: &KdlNode) -> Result<Vec<TypedVar>> {
        node.children()
            .into_iter()
            .flat_map(|d| d.nodes())
            .map(|var| {
                let name = self.var_name_decl(var)?;
                let ty_str = self.one_string(var, "type")?;
                let ty = self.ty_ref(&ty_str)?;
                self.no_children(var)?;
                Ok(TypedVar { name, ty })
            })
            .collect()
    }

    fn enum_variant_children(&mut self, node: &KdlNode) -> Result<Vec<EnumVariant>> {
        node.children()
            .into_iter()
            .flat_map(|d| d.nodes())
            .map(|var| {
                let name = var.name();
                let name = Spanned::new(name.value().to_owned(), *name.span());
                let entries = var.entries();
                let val = if let Some(e) = entries.get(0) {
                    Some(self.literal_expr(e)?)
                } else {
                    None
                };
                // TODO: deny any other members of `entries`
                self.no_children(var)?;
                Ok(EnumVariant { name, val })
            })
            .collect()
    }

    fn tagged_variant_children(&mut self, node: &KdlNode) -> Result<Vec<TaggedVariant>> {
        node.children()
            .into_iter()
            .flat_map(|d| d.nodes())
            .map(|var| {
                let name = var.name();
                let name = Spanned::new(name.value().to_owned(), *name.span());
                let fields = if var.children().is_some() {
                    Some(self.typed_var_children(var)?)
                } else {
                    None
                };
                Ok(TaggedVariant { name, fields })
            })
            .collect()
    }

    fn var_name_decl(&mut self, var: &KdlNode) -> Result<Option<Ident>> {
        let name = var.name();
        let name = if name.value() == "_" {
            None
        } else {
            Some(Spanned::new(name.value().to_owned(), *name.span()))
        };
        Ok(name)
    }

    fn expr_rhs(&mut self, node: &KdlNode, expr_start: usize) -> Result<Spanned<Expr>> {
        trace!("expr rhs");
        let expr = if let Ok(string) = self.string_at(node, "", expr_start) {
            if let Some((func, "")) = string.rsplit_once(':') {
                trace!("  call expr");
                let func = Spanned::new(func.to_owned(), Spanned::span(&string));
                let args = self.func_args(node, expr_start + 1)?;
                Expr::Call(CallExpr { func, args })
            } else if node.children().is_some() {
                trace!("  ctor expr");
                let ty = string;
                let vals = self.let_stmt_children(node)?;
                Expr::Ctor(CtorExpr { ty, vals })
            } else {
                trace!("  path expr");
                let mut parts = string.split('.');
                let var = Spanned::new(parts.next().unwrap().to_owned(), Spanned::span(&string));
                let path = parts
                    .map(|s| Spanned::new(s.to_owned(), Spanned::span(&string)))
                    .collect();
                Expr::Path(PathExpr { var, path })
            }
        } else if let Some(val) = node.entries().get(expr_start) {
            trace!("  literal expr");
            Expr::Literal(self.literal_expr(val)?)
        } else {
            return Err(KdlScriptParseError {
                message: format!("I thought there was supposed to be an expression after here?"),
                src: self.src.clone(),
                span: *node.span(),
                help: None,
            })?;
        };

        Ok(Spanned::new(expr, *node.span()))
    }

    fn smol_expr(&mut self, node: &KdlNode, expr_at: usize) -> Result<Spanned<Expr>> {
        trace!("smol expr");
        let expr = if let Ok(string) = self.string_at(node, "", expr_at) {
            if let Some((_func, "")) = string.rsplit_once(':') {
                return Err(KdlScriptParseError {
                    message: format!(
                        "Nested function calls aren't supported because this is a shitpost"
                    ),
                    src: self.src.clone(),
                    span: *node.span(),
                    help: None,
                })?;
            } else if node.children().is_some() {
                return Err(KdlScriptParseError {
                    message: format!(
                        "Ctors exprs can't be nested in function calls because this is a shitpost"
                    ),
                    src: self.src.clone(),
                    span: *node.span(),
                    help: None,
                })?;
            } else {
                trace!("  path expr");
                let mut parts = string.split('.');
                let var = Spanned::new(parts.next().unwrap().to_owned(), Spanned::span(&string));
                let path = parts
                    .map(|s| Spanned::new(s.to_owned(), Spanned::span(&string)))
                    .collect();
                Expr::Path(PathExpr { var, path })
            }
        } else if let Some(val) = node.entries().get(expr_at) {
            trace!("  literal expr");
            Expr::Literal(self.literal_expr(val)?)
        } else {
            return Err(KdlScriptParseError {
                message: format!("I thought there was supposed to be an expression after here?"),
                src: self.src.clone(),
                span: *node.span(),
                help: None,
            })?;
        };

        Ok(Spanned::new(expr, *node.span()))
    }

    fn ty_ref(&mut self, input: &Spanned<String>) -> Result<TyRef> {
        // TODO: validate this all better and write a proper parser
        if &**input == "()" {
            return Ok(TyRef::Empty);
        }
        if let Some(("", body)) = input.split_once('&') {
            let ty = Spanned::new(body.to_string(), Spanned::span(input));
            let ty = self.ty_ref(&ty)?;
            return Ok(TyRef::Ref(Box::new(ty)));
        }
        if let Some(("", body)) = input.split_once('[') {
            if let Some((body, "")) = body.rsplit_once(']') {
                if let Some((ty, size)) = body.rsplit_once(';') {
                    let size: u64 = size.parse().map_err(|_| KdlScriptParseError {
                        message: format!("invalid array size"),
                        src: self.src.clone(),
                        span: Spanned::span(input),
                        help: None,
                    })?;
                    let ty = Spanned::new(ty.to_string(), Spanned::span(input));
                    let ty = self.ty_ref(&ty)?;
                    return Ok(TyRef::Array(Box::new(ty), size));
                }
            }
        }
        return Ok(TyRef::Name(input.clone()));
    }

    fn func_args(&mut self, node: &KdlNode, expr_start: usize) -> Result<Vec<Spanned<Expr>>> {
        node.entries()[expr_start..]
            .iter()
            .enumerate()
            .map(|(idx, _e)| self.smol_expr(node, expr_start + idx))
            .collect()
    }

    fn let_stmt_children(&mut self, node: &KdlNode) -> Result<Vec<Spanned<LetStmt>>> {
        node.children()
            .into_iter()
            .flat_map(|d| d.nodes())
            .map(|var| {
                let name = self.var_name_decl(var)?;
                let expr = self.expr_rhs(var, 0)?;
                Ok(Spanned::new(LetStmt { var: name, expr }, *var.span()))
            })
            .collect()
    }

    fn literal_expr(&mut self, entry: &KdlEntry) -> Result<LiteralExpr> {
        if entry.name().is_some() {
            return Err(KdlScriptParseError {
                message: format!("Named values don't belong here, only literals"),
                src: self.src.clone(),
                span: *entry.span(),
                help: Some("try removing the name".to_owned()),
            })?;
        }

        let val = match entry.value() {
            kdl::KdlValue::RawString(_) | kdl::KdlValue::String(_) => {
                return Err(KdlScriptParseError {
                    message: format!("strings aren't supported literals"),
                    src: self.src.clone(),
                    span: *entry.span(),
                    help: None,
                })?;
            }
            kdl::KdlValue::Null => {
                return Err(KdlScriptParseError {
                    message: format!("nulls aren't supported literals"),
                    src: self.src.clone(),
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
}

pub use runnable::*;
mod runnable {
    use super::*;

    #[derive(Debug, Clone)]
    pub enum Stmt {
        Let(LetStmt),
        Return(ReturnStmt),
        Print(PrintStmt),
    }

    #[derive(Debug, Clone)]
    pub struct LetStmt {
        pub var: Option<Ident>,
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
        pub func: Ident,
        pub args: Vec<Spanned<Expr>>,
    }

    #[derive(Debug, Clone)]
    pub struct PathExpr {
        pub var: Ident,
        pub path: Vec<Ident>,
    }

    #[derive(Debug, Clone)]
    pub struct CtorExpr {
        pub ty: Ident,
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

    impl ParsedProgram {
        pub(crate) fn add_builtins(&mut self) -> Result<()> {
            // Add builtins jankily
            self.funcs.insert(
                Spanned::from(String::from("+")),
                FuncDecl {
                    name: Spanned::from(String::from("+")),
                    inputs: vec![
                        TypedVar {
                            name: Some(Spanned::from(String::from("lhs"))),
                            ty: TyRef::Name(Spanned::from(String::from("i64"))),
                        },
                        TypedVar {
                            name: Some(Spanned::from(String::from("rhs"))),
                            ty: TyRef::Name(Spanned::from(String::from("i64"))),
                        },
                    ],
                    outputs: vec![TypedVar {
                        name: Some(Spanned::from(String::from("out"))),
                        ty: TyRef::Name(Spanned::from(String::from("i64"))),
                    }],
                    attrs: vec![],
                    body: vec![],
                },
            );

            Ok(())
        }
    }
}
