//! Tests for the big-endian join order: how the second register's value at a
//! RETURN is classified, on a hand-built `Funcdata`. The end-to-end witnesses
//! are the compiled round trips in `kuna-cli/tests/decompile_all_cli.rs`.

use super::*;

use std::rc::Rc;

use kuna_base::address::Address;
use kuna_base::space::{addrspace_flags, spacetype, AddrSpace, AddrSpaceManager, ConstantSpace, UniqueSpace};

use crate::context::{ArchContext, TypeOp};

/// The first return register (the high word) and the second (the low word).
const HI: u64 = 0x10;
const LO: u64 = 0x14;

fn build_fd() -> Funcdata {
    let mut m = AddrSpaceManager::new();
    m.insert_space(Rc::new(ConstantSpace::new())).unwrap();
    m.insert_space(Rc::new(UniqueSpace::new(1, 0, false))).unwrap();
    m.insert_space(Rc::new(AddrSpace::new(
        spacetype::IPTR_PROCESSOR,
        "register",
        true,
        4,
        1,
        2,
        addrspace_flags::hasphysical,
        1,
        1,
    )))
    .unwrap();
    let glb = Rc::new(ArchContext::new(m));
    let reg = Rc::clone(glb.manage().get_space_by_name("register").unwrap());
    Funcdata::new("func", "func", glb, Address::new(reg, 0x1000), 0x1000_0000, 0x40).unwrap()
}

fn at(fd: &Funcdata, off: u64) -> Address {
    Address::new(Rc::clone(fd.get_arch().manage().get_space_by_name("register").unwrap()), off)
}

fn pair(fd: &Funcdata) -> Pair {
    Pair { lo_slot: 2, hi_slot: 1, lo_size: 4, own: at(fd, LO) }
}

/// The function's input in the register at `off`.
fn input(fd: &mut Funcdata, off: u64) -> VarnodeId {
    let a = at(fd, off);
    let vn = fd.new_varnode(4, &a, None);
    fd.set_input_varnode(vn).expect("input")
}

/// A live `<opc>(inputs)` in the function's one basic block.
fn op(fd: &mut Funcdata, opc: OpCode, inputs: &[VarnodeId]) -> OpId {
    let a = at(fd, 0x2000);
    let op = fd.new_op(inputs.len() as int4, a);
    fd.op_set_opcode(op, TypeOp::new(opc, 0, format!("{opc:?}")));
    for (i, &vn) in inputs.iter().enumerate() {
        fd.op_set_input(op, vn, i as int4).expect("input");
    }
    let root = fd.bblocks_ref().root.expect("bblocks root");
    let bl = match fd.bblocks_ref().block(root).get_size() {
        0 => fd.bblocks_mut().new_block_basic(root),
        _ => fd.bblocks_ref().block(root).get_block(0),
    };
    fd.op_insert(op, bl, None);
    op
}

/// `out = <opc>(inputs)` with `out` a `size`-byte register at `off`.
fn def(fd: &mut Funcdata, opc: OpCode, inputs: &[VarnodeId], off: u64, size: int4) -> VarnodeId {
    let o = op(fd, opc, inputs);
    let a = at(fd, off);
    fd.new_varnode_out(size, &a, o).expect("output")
}

fn k(fd: &mut Funcdata, v: u64) -> VarnodeId {
    fd.new_constant(4, v)
}

fn ret(fd: &mut Funcdata, high: VarnodeId, low: VarnodeId) -> OpId {
    let a = k(fd, 0);
    op(fd, OpCode::CPUI_RETURN, &[a, high, low])
}

fn class_of(fd: &mut Funcdata, high: VarnodeId, low: VarnodeId) -> LowWord {
    let r = ret(fd, high, low);
    let p = pair(fd);
    classify(fd, r, &p)
}

#[test]
fn a_product_only_the_return_reads_is_returned() {
    let mut fd = build_fd();
    let (a, b) = (input(&mut fd, 0x20), input(&mut fd, 0x24));
    let lo = def(&mut fd, OpCode::CPUI_INT_MULT, &[a, b], LO, 4);
    let hi = def(&mut fd, OpCode::CPUI_INT_MULT, &[a, b], HI, 4);
    assert_eq!(class_of(&mut fd, hi, lo), LowWord::Returned);
}

#[test]
fn a_literal_zero_proves_nothing() {
    let mut fd = build_fd();
    let a = input(&mut fd, 0x20);
    let zero = k(&mut fd, 0);
    let lo = def(&mut fd, OpCode::CPUI_COPY, &[zero], LO, 4);
    let hi = def(&mut fd, OpCode::CPUI_INT_2COMP, &[a], HI, 4);
    assert_eq!(class_of(&mut fd, hi, lo), LowWord::Zero, "clang -O0 leaves `addiu $3,$zero,0` in an int function");
}

#[test]
fn a_literal_zero_built_in_two_steps_is_still_zero() {
    let mut fd = build_fd();
    let a = input(&mut fd, 0x20);
    let (z1, z2) = (k(&mut fd, 0), k(&mut fd, 0));
    let lo = def(&mut fd, OpCode::CPUI_INT_ADD, &[z1, z2], LO, 4);
    let hi = def(&mut fd, OpCode::CPUI_INT_2COMP, &[a], HI, 4);
    assert_eq!(class_of(&mut fd, hi, lo), LowWord::Zero);
}

#[test]
fn a_nonzero_literal_only_the_return_reads_is_returned() {
    let mut fd = build_fd();
    let (ten, zero) = (k(&mut fd, 10), k(&mut fd, 0));
    let lo = def(&mut fd, OpCode::CPUI_COPY, &[ten], LO, 4);
    let hi = def(&mut fd, OpCode::CPUI_COPY, &[zero], HI, 4);
    assert_eq!(class_of(&mut fd, hi, lo), LowWord::Returned, "`return 10LL`");
}

#[test]
fn the_register_left_as_it_arrived_is_a_leftover() {
    let mut fd = build_fd();
    let a = input(&mut fd, 0x20);
    let lo = input(&mut fd, LO);
    let hi = def(&mut fd, OpCode::CPUI_INT_2COMP, &[a], HI, 4);
    assert_eq!(class_of(&mut fd, hi, lo), LowWord::Entry);
}

#[test]
fn the_register_moved_away_and_back_is_returned() {
    let mut fd = build_fd();
    let entry = input(&mut fd, LO);
    let saved = def(&mut fd, OpCode::CPUI_COPY, &[entry], 0x30, 4);
    let lo = def(&mut fd, OpCode::CPUI_COPY, &[saved], LO, 4);
    let zero = k(&mut fd, 0);
    let hi = def(&mut fd, OpCode::CPUI_COPY, &[zero], HI, 4);
    assert_eq!(
        class_of(&mut fd, hi, lo),
        LowWord::Returned,
        "ARM's `mov r4,r1; bl ext; mov r1,r4` carries the argument into the low word on purpose",
    );
}

#[test]
fn a_value_also_stored_is_scratch() {
    let mut fd = build_fd();
    let (a, b, p) = (input(&mut fd, 0x20), input(&mut fd, 0x24), input(&mut fd, 0x28));
    let lo = def(&mut fd, OpCode::CPUI_INT_ADD, &[a, b], LO, 4);
    let space = k(&mut fd, 0);
    op(&mut fd, OpCode::CPUI_STORE, &[space, p, lo]);
    let hi = def(&mut fd, OpCode::CPUI_INT_MULT, &[a, b], HI, 4);
    assert_eq!(class_of(&mut fd, hi, lo), LowWord::Scratch);
}

#[test]
fn a_value_used_as_an_address_is_scratch() {
    let mut fd = build_fd();
    let (a, b) = (input(&mut fd, 0x20), input(&mut fd, 0x24));
    let lo = def(&mut fd, OpCode::CPUI_INT_ADD, &[a, b], LO, 4);
    let space = k(&mut fd, 0);
    let hi = def(&mut fd, OpCode::CPUI_LOAD, &[space, lo], HI, 4);
    assert_eq!(class_of(&mut fd, hi, lo), LowWord::Scratch, "`__pgetc` keeps a pointer it read through in %i1");
}

#[test]
fn a_value_the_first_register_is_computed_from_is_scratch() {
    let mut fd = build_fd();
    let (a, b) = (input(&mut fd, 0x20), input(&mut fd, 0x24));
    let lo = def(&mut fd, OpCode::CPUI_INT_MULT, &[a, b], LO, 4);
    let hi = def(&mut fd, OpCode::CPUI_INT_ADD, &[lo, a], HI, 4);
    assert_eq!(class_of(&mut fd, hi, lo), LowWord::Scratch);
}

#[test]
fn the_same_value_in_both_registers_is_scratch() {
    let mut fd = build_fd();
    let (a, b) = (input(&mut fd, 0x20), input(&mut fd, 0x24));
    let lo = def(&mut fd, OpCode::CPUI_INT_MULT, &[a, b], LO, 4);
    let hi = def(&mut fd, OpCode::CPUI_COPY, &[lo], HI, 4);
    assert_eq!(class_of(&mut fd, hi, lo), LowWord::Scratch, "either join reads the same value");
}

#[test]
fn the_sign_of_the_low_word_may_build_the_first_register() {
    let mut fd = build_fd();
    let (a, b) = (input(&mut fd, 0x20), input(&mut fd, 0x24));
    let lo = def(&mut fd, OpCode::CPUI_INT_MULT, &[a, b], LO, 4);
    let bits = k(&mut fd, 31);
    let hi = def(&mut fd, OpCode::CPUI_INT_SRIGHT, &[lo, bits], HI, 4);
    assert_eq!(class_of(&mut fd, hi, lo), LowWord::Returned, "`(long long)x`");
}

#[test]
fn a_carry_out_of_the_low_word_may_build_the_first_register() {
    let mut fd = build_fd();
    let (ah, al) = (input(&mut fd, 0x20), input(&mut fd, 0x24));
    let one = k(&mut fd, 1);
    let lo = def(&mut fd, OpCode::CPUI_INT_ADD, &[al, one], LO, 4);
    let carry = def(&mut fd, OpCode::CPUI_INT_LESS, &[lo, al], 0x40, 1);
    let widened = def(&mut fd, OpCode::CPUI_INT_ZEXT, &[carry], 0x44, 4);
    let hi = def(&mut fd, OpCode::CPUI_INT_ADD, &[ah, widened], HI, 4);
    assert_eq!(class_of(&mut fd, hi, lo), LowWord::Returned, "MIPS adds a `long long` with `sltu`");
}

#[test]
fn the_halves_of_one_wide_value_are_wide() {
    let mut fd = build_fd();
    let (space, p) = (k(&mut fd, 0), input(&mut fd, 0x20));
    let whole = def(&mut fd, OpCode::CPUI_LOAD, &[space, p], 0x50, 8);
    let (z, four) = (k(&mut fd, 0), k(&mut fd, 4));
    let lo = def(&mut fd, OpCode::CPUI_SUBPIECE, &[whole, z], LO, 4);
    let hi = def(&mut fd, OpCode::CPUI_SUBPIECE, &[whole, four], HI, 4);
    let space2 = k(&mut fd, 0);
    op(&mut fd, OpCode::CPUI_STORE, &[space2, p, lo]);
    assert_eq!(class_of(&mut fd, hi, lo), LowWord::Wide, "an 8-byte value is returned whole, whatever else reads it");
}

#[test]
fn swapped_halves_are_not_one_value() {
    let mut fd = build_fd();
    let (space, p) = (k(&mut fd, 0), input(&mut fd, 0x20));
    let whole = def(&mut fd, OpCode::CPUI_LOAD, &[space, p], 0x50, 8);
    let (z, four) = (k(&mut fd, 0), k(&mut fd, 4));
    let hi = def(&mut fd, OpCode::CPUI_SUBPIECE, &[whole, z], HI, 4);
    let lo = def(&mut fd, OpCode::CPUI_SUBPIECE, &[whole, four], LO, 4);
    assert!(!halves_of_one_value(&fd, lo, hi, 4));
}

#[test]
fn literal_values_fold_the_way_sparc_builds_them() {
    let mut fd = build_fd();
    let (hi22, lo10) = (k(&mut fd, 0x9e377800), k(&mut fd, 0x1b9));
    let sethi = def(&mut fd, OpCode::CPUI_COPY, &[hi22], 0x60, 4);
    let or = def(&mut fd, OpCode::CPUI_INT_OR, &[sethi, lo10], 0x64, 4);
    assert_eq!(literal_value(&fd, or, LITERAL_DEPTH), Some(0x9e3779b9));
    let (a, lo10b) = (input(&mut fd, 0x20), k(&mut fd, 0x1b9));
    let not_literal = def(&mut fd, OpCode::CPUI_INT_OR, &[a, lo10b], 0x68, 4);
    assert_eq!(literal_value(&fd, not_literal, LITERAL_DEPTH), None);
}

#[test]
fn a_literal_multiplied_into_the_first_register_is_scratch() {
    let mut fd = build_fd();
    let a = input(&mut fd, 0x20);
    let magic = k(&mut fd, 0xb6db6db7);
    let lo = def(&mut fd, OpCode::CPUI_COPY, &[magic], LO, 4);
    let (wide_lo, wide_a) = (def(&mut fd, OpCode::CPUI_INT_ZEXT, &[lo], 0x50, 8), def(&mut fd, OpCode::CPUI_INT_ZEXT, &[a], 0x58, 8));
    let product = def(&mut fd, OpCode::CPUI_INT_MULT, &[wide_lo, wide_a], 0x60, 8);
    let four = k(&mut fd, 4);
    let hi = def(&mut fd, OpCode::CPUI_SUBPIECE, &[product, four], HI, 4);
    assert_eq!(class_of(&mut fd, hi, lo), LowWord::Scratch, "SPARC's `umul` by a divisor's magic number left in %i1");
}

#[test]
fn the_high_half_of_the_low_words_extension_may_build_the_first_register() {
    let mut fd = build_fd();
    let (a, b) = (input(&mut fd, 0x20), input(&mut fd, 0x24));
    let lo = def(&mut fd, OpCode::CPUI_INT_MULT, &[a, b], LO, 4);
    let wide = def(&mut fd, OpCode::CPUI_INT_SEXT, &[lo], 0x50, 8);
    let four = k(&mut fd, 4);
    let hi = def(&mut fd, OpCode::CPUI_SUBPIECE, &[wide, four], HI, 4);
    assert_eq!(class_of(&mut fd, hi, lo), LowWord::Returned, "`(long long)(a * b)`");
}

#[test]
fn a_register_pair_heritage_split_is_followed_to_the_entry_value() {
    let mut fd = build_fd();
    let a = input(&mut fd, 0x20);
    let at_pair = at(&fd, HI);
    let pair_in = fd.new_varnode(8, &at_pair, None);
    let pair_in = fd.set_input_varnode(pair_in).expect("input");
    let zero = k(&mut fd, 0);
    let lo = def(&mut fd, OpCode::CPUI_SUBPIECE, &[pair_in, zero], LO, 4);
    let hi = def(&mut fd, OpCode::CPUI_INT_2COMP, &[a], HI, 4);
    assert_eq!(
        class_of(&mut fd, hi, lo),
        LowWord::Entry,
        "the low half of the %i0:%i1 pair as it arrived is the second register's entry value",
    );
}

#[test]
fn a_product_split_by_a_shift_is_one_wide_value() {
    let mut fd = build_fd();
    let (a, b) = (input(&mut fd, 0x20), input(&mut fd, 0x24));
    let (wa, wb) = (def(&mut fd, OpCode::CPUI_INT_SEXT, &[a], 0x50, 8), def(&mut fd, OpCode::CPUI_INT_SEXT, &[b], 0x58, 8));
    let product = def(&mut fd, OpCode::CPUI_INT_MULT, &[wa, wb], 0x60, 8);
    let (z1, z2, bits) = (k(&mut fd, 0), k(&mut fd, 0), k(&mut fd, 32));
    let lo = def(&mut fd, OpCode::CPUI_SUBPIECE, &[product, z1], LO, 4);
    let shifted = def(&mut fd, OpCode::CPUI_INT_RIGHT, &[product, bits], 0x68, 8);
    let hi = def(&mut fd, OpCode::CPUI_SUBPIECE, &[shifted, z2], HI, 4);
    assert_eq!(class_of(&mut fd, hi, lo), LowWord::Wide, "MIPS `mult $4,$5; mflo $3; mfhi $2`");
}
