//! Type System!

use std::collections::HashMap;

use crate::parse::*;
use crate::spanned::*;
use crate::Compiler;
use crate::Result;

#[derive(Debug)]
pub struct TypedProgram {
    tcx: TyCtx,
    funcs: Vec<Func>,
}

pub type TyIdx = usize;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Ty {
    Primitive(PrimitiveTy),
    Struct(StructTy),
    Union(UnionTy),
    Enum(EnumTy),
    Tagged(TaggedTy),
    Alias(AliasTy),
    Pun(PunTy),
    Array(ArrayTy),
    Ref(RefTy),
    /// Empty tuple ()
    Empty,
}

#[derive(Debug, Clone)]
pub struct Func {
    name: Ident,
    inputs: Vec<Arg>,
    outputs: Vec<Arg>,
    attrs: Vec<Attr>,
    body: (),
}

#[derive(Debug, Clone)]
pub struct Arg {
    name: Ident,
    ty: TyIdx,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum PrimitiveTy {
    I8,
    I16,
    I32,
    I64,
    I128,
    I256,
    U8,
    U16,
    U32,
    U64,
    U128,
    U256,
    F16,
    F32,
    F64,
    F128,
    Bool,
    /// An opaque pointer (like `void*`)
    Ptr,
}

/// The Ty of a Struct.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct StructTy {
    pub name: Ident,
    pub fields: Vec<FieldTy>,
    pub attrs: Vec<Attr>,
}

/// The Ty of an Union.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct UnionTy {
    pub name: Ident,
    pub fields: Vec<FieldTy>,
    pub attrs: Vec<Attr>,
}

/// The Ty of an Enum.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct EnumTy {
    pub name: Ident,
    pub variants: Vec<EnumVariantTy>,
    pub attrs: Vec<Attr>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct EnumVariantTy {
    pub name: Ident,
    // pub val: LiteralExpr,
}

/// The Ty of a Tagged.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TaggedTy {
    pub name: Ident,
    pub variants: Vec<TaggedVariantTy>,
    pub attrs: Vec<Attr>,
}

/// The Ty of a Tagged Union.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TaggedVariantTy {
    pub name: Ident,
    pub fields: Option<Vec<FieldTy>>,
}

/// The Ty of an Alias.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct AliasTy {
    pub name: Ident,
    pub real: TyIdx,
    pub attrs: Vec<Attr>,
}

/// The Ty of a specific field of a struct.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct FieldTy {
    pub ident: Ident,
    pub ty: TyIdx,
}

/// The Ty of an Array.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ArrayTy {
    pub elem_ty: TyIdx,
    pub len: u64,
}

/// The Ty of a Reference (transparent pointer).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct RefTy {
    pub pointee_ty: TyIdx,
}

/// The Ty of a Struct.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PunTy {
    pub name: Ident,
    pub blocks: Vec<PunBlockTy>,
    pub attrs: Vec<Attr>,
}

/// The Ty of a Struct.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PunBlockTy {
    pub selector: PunSelector,
    pub real: TyIdx,
}

/// Information on all the types.
///
/// The key function of TyCtx is to `memoize` all parsed types (TyName) into
/// type ids (TyIdx), to enable correct type comparison. Two types are equal
/// *if and only if* they have the same TyIdx.
///
/// This is necessary because *nominal* types (TyName::Named, i.e. structs) can
/// be messy due to shenanigans like captures/scoping/shadowing/inference. Types
/// may refer to names that are out of scope, and two names that are equal
/// (as strings) may not actually refer to the same type declaration.
///
/// To handle this, whenever a new named type is declared ([push_struct_decl][]),
/// we generate a unique type id (TyIdx) for it. Then whenever we encounter
/// a reference to a Named type, we lookup the currently in scope TyIdx for that
/// name, and use that instead. Named type scoping is managed by `envs`.
///
/// Replacing type names with type ids requires a change of representation,
/// which is why we have [Ty][]. A Ty is the *structure* of a type with all subtypes
/// resolved to TyIdx's (e.g. a field of a tuple, the return type of a function).
/// For convenience, non-typing metadata may also be stored in a Ty.
///
/// So a necessary intermediate step of converting TyName to a TyIdx is to first
/// convert it to a Ty. This intermediate value is stored in `tys`.
/// If you have a TyIdx, you can get its Ty with [realize_ty][]. This lets you
/// e.g. check if a value being called is actually a Func, and if it is,
/// what the type ids of its arguments/return types are.
///
/// `ty_map` stores all the *structural* Tys we've seen before (everything that
/// *isn't* TyName::Named), ensuring two structural types have the same TyIdx.
/// i.e. `(Bool, Int)` will have the same TyIdx everywhere it occurs.
#[derive(Debug)]
pub struct TyCtx {
    /// The list of every known type.
    ///
    /// These are the "canonical" copies of each type. Types are
    /// registered here via `memoize`, which returns a TyIdx into
    /// this array.
    ///
    /// Types should be compared by checking if they have the same
    /// TyIdx. This allows you to properly compare nominal types
    /// in the face of shadowing and similar situations.
    pub tys: Vec<Ty>,

    /// Mappings from structural types we've seen to type indices.
    ///
    /// This is used to get the canonical TyIdx of a structural type
    /// (including builtin primitives).
    ///
    /// Nominal types (structs) are stored in `envs`, because they
    /// go in and out of scope.
    ty_map: HashMap<Ty, TyIdx>,

    /// Scoped type info, reflecting the fact that struct definitions
    /// and variables come in and out of scope.
    ///
    /// These values are "cumulative", so type names and variables
    /// should be looked up by searching backwards in this array.
    ///
    /// If nothing is found, that type name / variable name is undefined
    /// at this point in the program.
    envs: Vec<CheckEnv>,
}

/// Information about types for a specific scope.
#[derive(Debug)]
struct CheckEnv {
    /// The struct definitions and TyIdx's
    tys: HashMap<Ident, TyIdx>,
}

pub fn typeck(comp: &mut Compiler, parsed: &ParsedProgram) -> Result<TypedProgram> {
    let mut tcx = TyCtx {
        tys: vec![],
        ty_map: HashMap::new(),
        envs: vec![],
    };

    // Add global builtins
    tcx.envs.push(CheckEnv {
        tys: HashMap::new(),
    });
    tcx.add_builtins();

    // Put user-defined types in a separate scope just to be safe
    tcx.envs.push(CheckEnv {
        tys: HashMap::new(),
    });

    // Add all the user defined types
    for (ty_name, _ty_decl) in &parsed.tys {
        let _ty_idx = tcx.push_nominal_decl_incomplete(ty_name.clone());
    }
    for (ty_name, ty_decl) in &parsed.tys {
        tcx.complete_nominal_decl(ty_name, ty_decl);
    }

    let funcs = parsed
        .funcs
        .iter()
        .map(|(_func_name, func_decl)| {
            let inputs = func_decl
                .inputs
                .iter()
                .map(|var| {
                    let name = var.name.clone().expect("TODO: impl optional names");
                    let ty = tcx.memoize_ty(&var.ty);
                    Arg { name, ty }
                })
                .collect();
            let outputs = func_decl
                .outputs
                .iter()
                .map(|var| {
                    let name = var.name.clone().expect("TODO: impl optional names");
                    let ty = tcx.memoize_ty(&var.ty);
                    Arg { name, ty }
                })
                .collect();

            let name = func_decl.name.clone();
            let attrs = func_decl.attrs.clone();
            Func {
                name,
                inputs,
                outputs,
                attrs,
                body: (),
            }
        })
        .collect();

    Ok(TypedProgram { tcx, funcs })
}

impl TyCtx {
    fn add_builtins(&mut self) {
        let builtins = [
            ("i8", Ty::Primitive(PrimitiveTy::I8)),
            ("i16", Ty::Primitive(PrimitiveTy::I16)),
            ("i32", Ty::Primitive(PrimitiveTy::I32)),
            ("i64", Ty::Primitive(PrimitiveTy::I64)),
            ("i128", Ty::Primitive(PrimitiveTy::I128)),
            ("i256", Ty::Primitive(PrimitiveTy::I256)),
            ("u8", Ty::Primitive(PrimitiveTy::U8)),
            ("u16", Ty::Primitive(PrimitiveTy::U16)),
            ("u32", Ty::Primitive(PrimitiveTy::U32)),
            ("u64", Ty::Primitive(PrimitiveTy::U64)),
            ("u128", Ty::Primitive(PrimitiveTy::U128)),
            ("u256", Ty::Primitive(PrimitiveTy::U256)),
            ("f16", Ty::Primitive(PrimitiveTy::F16)),
            ("f32", Ty::Primitive(PrimitiveTy::F32)),
            ("f64", Ty::Primitive(PrimitiveTy::F64)),
            ("f128", Ty::Primitive(PrimitiveTy::F128)),
            ("bool", Ty::Primitive(PrimitiveTy::Bool)),
            ("ptr", Ty::Primitive(PrimitiveTy::Ptr)),
        ];

        for (ty_name, ty) in builtins {
            let ty_idx = self.tys.len();
            self.tys.push(ty);
            self.envs
                .last_mut()
                .unwrap()
                .tys
                .insert(Spanned::from(ty_name.to_owned()), ty_idx);
        }
    }
    /// Register a new nominal struct in this scope.
    ///
    /// This creates a valid TyIdx for the type, but the actual Ty
    /// while be garbage (Ty::Empty arbitrarily) and needs to be
    /// filled in properly with [`TyCtx::complete_nominal_decl`][].
    ///
    /// This two-phase system is necessary to allow nominal types to
    /// be unordered or self-referential.
    fn push_nominal_decl_incomplete(&mut self, ty_name: Ident) -> TyIdx {
        let ty_idx = self.tys.len();
        let dummy_ty = Ty::Empty;
        self.tys.push(dummy_ty);
        self.envs.last_mut().unwrap().tys.insert(ty_name, ty_idx);
        ty_idx
    }

    fn complete_nominal_decl(&mut self, ty_name: &Ident, ty_decl: &TyDecl) {
        let ty_idx = self
            .resolve_nominal_ty(ty_name)
            .expect("completing a nominal ty that hasn't been decl'd");
        let ty = self.memoize_nominal_parts(ty_decl);
        self.tys[ty_idx] = ty;
    }

    fn memoize_nominal_parts(&mut self, ty_decl: &TyDecl) -> Ty {
        match ty_decl {
            TyDecl::Struct(decl) => {
                let fields = decl
                    .fields
                    .iter()
                    .map(|f| FieldTy {
                        ident: f.name.clone().expect("TODO: implement unnamed fields"),
                        ty: self.memoize_ty(&f.ty),
                    })
                    .collect::<Vec<_>>();
                Ty::Struct(StructTy {
                    name: decl.name.clone(),
                    fields,
                    attrs: decl.attrs.clone(),
                })
            }
            TyDecl::Union(decl) => {
                let fields = decl
                    .fields
                    .iter()
                    .map(|f| FieldTy {
                        ident: f.name.clone().expect("TODO: implement unnamed fields"),
                        ty: self.memoize_ty(&f.ty),
                    })
                    .collect::<Vec<_>>();
                Ty::Union(UnionTy {
                    name: decl.name.clone(),
                    fields,
                    attrs: decl.attrs.clone(),
                })
            }
            TyDecl::Enum(decl) => {
                let variants = decl
                    .variants
                    .iter()
                    .map(|v| EnumVariantTy {
                        name: v.name.clone(),
                    })
                    .collect::<Vec<_>>();
                Ty::Enum(EnumTy {
                    name: decl.name.clone(),
                    variants,
                    attrs: decl.attrs.clone(),
                })
            }
            TyDecl::Tagged(decl) => {
                let variants = decl
                    .variants
                    .iter()
                    .map(|v| TaggedVariantTy {
                        name: v.name.clone(),
                        fields: v.fields.as_ref().map(|fields| {
                            fields
                                .iter()
                                .map(|f| FieldTy {
                                    ident: f.name.clone().expect("TODO: implement unnamed fields"),
                                    ty: self.memoize_ty(&f.ty),
                                })
                                .collect::<Vec<_>>()
                        }),
                    })
                    .collect::<Vec<_>>();
                Ty::Tagged(TaggedTy {
                    name: decl.name.clone(),
                    variants,
                    attrs: decl.attrs.clone(),
                })
            }
            TyDecl::Alias(decl) => {
                let real_ty = self.memoize_ty(&decl.alias);
                Ty::Alias(AliasTy {
                    name: decl.name.clone(),
                    real: real_ty,
                    attrs: decl.attrs.clone(),
                })
            }
            TyDecl::Pun(decl) => {
                let blocks = decl
                    .blocks
                    .iter()
                    .map(|block| {
                        // !!! If this ever becomes fallible we'll want a proper stack guard to pop!
                        self.envs.push(CheckEnv {
                            tys: HashMap::new(),
                        });
                        let real_decl = &block.decl;
                        let real = self.push_nominal_decl_incomplete(decl.name.clone());
                        self.complete_nominal_decl(&decl.name, &real_decl);
                        self.envs.pop();

                        PunBlockTy {
                            selector: block.selector.clone(),
                            real,
                        }
                    })
                    .collect();

                Ty::Pun(PunTy {
                    name: decl.name.clone(),
                    blocks,
                    attrs: decl.attrs.clone(),
                })
            }
        }
    }

    /// Resolve the type id (TyIdx) associated with a nominal type (struct name),
    /// at this point in the program.
    fn resolve_nominal_ty(&mut self, ty_name: &str) -> Option<TyIdx> {
        for (_depth, env) in self.envs.iter_mut().rev().enumerate() {
            if let Some(ty) = env.tys.get(ty_name) {
                return Some(*ty);
            }
        }
        None
    }

    /// Converts a TyName (parsed type) into a TyIdx (type id).
    ///
    /// All TyNames in the program must be memoized, as this is the only reliable
    /// way to do type comparisons. See the top level docs of TyIdx for details.
    fn memoize_ty(&mut self, ty_ref: &TyRef) -> TyIdx {
        match ty_ref {
            TyRef::Empty => self.memoize_inner(Ty::Empty),
            TyRef::Ref(pointee_ty_ref) => {
                let pointee_ty = self.memoize_ty(pointee_ty_ref);
                self.memoize_inner(Ty::Ref(RefTy { pointee_ty }))
            }
            TyRef::Array(elem_ty_ref, len) => {
                let elem_ty = self.memoize_ty(elem_ty_ref);
                self.memoize_inner(Ty::Array(ArrayTy { elem_ty, len: *len }))
            }
            TyRef::Name(name) => {
                // Nominal types take a separate path because they're scoped
                if let Some(ty_idx) = self.resolve_nominal_ty(name) {
                    ty_idx
                } else {
                    // FIXME: rejig this so the line info is better
                    panic!("Compile Error: use of undefined type name: {}", name);
                    /*
                    program.error(
                        format!("Compile Error: use of undefined type name: {}", name),
                        Span {
                            start: addr(program.input),
                            end: addr(program.input),
                        },
                    )
                     */
                }
            }
        }
    }

    /// Converts a Ty (structural type with all subtypes resolved) into a TyIdx (type id).
    fn memoize_inner(&mut self, ty: Ty) -> TyIdx {
        if let Some(idx) = self.ty_map.get(&ty) {
            *idx
        } else {
            let ty1 = ty.clone();
            let ty2 = ty;
            let idx = self.tys.len();

            self.ty_map.insert(ty1, idx);
            self.tys.push(ty2);
            idx
        }
    }

    /// Get the type-structure (Ty) associated with this type id (TyIdx).
    pub fn realize_ty(&self, ty: TyIdx) -> &Ty {
        self.tys
            .get(ty)
            .expect("Internal Compiler Error: invalid TyIdx")
    }

    /*
    pub fn pointee_ty(&self, ty: TyIdx) -> TyIdx {
        if let Ty::TypedPtr(pointee) = self.realize_ty(ty) {
            *pointee
        } else {
            unreachable!("expected typed to be pointer");
        }
    }
     */

    /// Stringify a type.
    pub fn format_ty(&self, ty: TyIdx) -> String {
        match self.realize_ty(ty) {
            Ty::Primitive(prim) => format!("{:?}", prim).to_lowercase(),
            Ty::Empty => format!("()"),
            Ty::Struct(decl) => format!("{}", decl.name),
            Ty::Enum(decl) => format!("{}", decl.name),
            Ty::Tagged(decl) => format!("{}", decl.name),
            Ty::Union(decl) => format!("{}", decl.name),
            Ty::Alias(decl) => format!("{}", decl.name),
            Ty::Pun(decl) => format!("{}", decl.name),
            Ty::Array(array_ty) => {
                let inner = self.format_ty(array_ty.elem_ty);
                format!("[{}; {}]", inner, array_ty.len)
            }
            Ty::Ref(ref_ty) => {
                let inner = self.format_ty(ref_ty.pointee_ty);
                format!("&{}", inner)
            }
        }
    }
}
