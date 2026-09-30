//! Tests for the uncomputed-return-trial rule.
//!
//! These pin the *classification* — which value shapes count as computed — on a
//! hand-built `Funcdata`. The end-to-end witness (a real binary whose `main`
//! stops returning a 16-byte phantom) lives in
//! `kuna-console/tests/verify_return_uncomputed.rs`.

use super::*;

use std::rc::Rc;

use kuna_base::address::Address;
use kuna_base::types::int4;
use kuna_base::space::{
    addrspace_flags, spacetype, AddrSpace, AddrSpaceManager, ConstantSpace, UniqueSpace,
};

use crate::context::{ArchContext, TypeOp};

fn computes_a_value(data: &Funcdata, vn: VarnodeId, depth: u32) -> bool {
    computes_from(data, vn, depth, None)
}

fn build_manager() -> AddrSpaceManager {
    let mut m = AddrSpaceManager::new();
    m.insert_space(Rc::new(ConstantSpace::new())).unwrap();
    m.insert_space(Rc::new(UniqueSpace::new(1, 0, false))).unwrap();
    m.insert_space(Rc::new(AddrSpace::new(
        spacetype::IPTR_PROCESSOR,
        "ram",
        false,
        8,
        1,
        2,
        addrspace_flags::hasphysical,
        1,
        1,
    )))
    .unwrap();
    m
}

fn build_fd() -> Funcdata {
    let manage = build_manager();
    let glb = Rc::new(ArchContext::new(manage));
    let ram = Rc::clone(glb.manage().get_space_by_name("ram").unwrap());
    let addr = Address::new(ram, 0x1000);
    Funcdata::new("func", "func", glb, addr, 0x1000_0000, 0x40).unwrap()
}

fn ram(fd: &Funcdata) -> Rc<AddrSpace> {
    Rc::clone(fd.get_arch().manage().get_space_by_name("ram").unwrap())
}

/// A Varnode at a ram address with no defining op — a function input, or a
/// location the function only ever reads.
fn unwritten(fd: &mut Funcdata, off: u64, sz: int4) -> VarnodeId {
    let r = ram(fd);
    fd.new_varnode(sz, &Address::new(r, off), None)
}

/// Build `out = <opc>(inputs...)` at a fresh ram address and return `out`.
fn mk_def(fd: &mut Funcdata, opc: OpCode, inputs: &[VarnodeId], out_off: u64) -> VarnodeId {
    let r = ram(fd);
    let op = fd.new_op(inputs.len() as int4, Address::new(Rc::clone(&r), out_off));
    fd.obank_mut().change_opcode(op, TypeOp::new(opc, 0, format!("{opc:?}")));
    for (i, &vn) in inputs.iter().enumerate() {
        fd.op_set_input(op, vn, i as int4).expect("wire input");
    }
    fd.new_varnode_out(8, &Address::new(r, out_off), op).expect("varnode out")
}

#[test]
fn a_constant_is_a_computed_return_value() {
    let mut fd = build_fd();
    let k = fd.new_constant(8, 7);
    assert!(
        computes_a_value(&fd, k, 0),
        "`return 0;` is a real return — a literal must never be dropped",
    );
}

#[test]
fn an_unwritten_varnode_is_not_a_computed_return_value() {
    let mut fd = build_fd();
    let vn = unwritten(&mut fd, 0x2000, 8);
    assert!(
        !computes_a_value(&fd, vn, 0),
        "an unwritten Varnode carries whatever the caller left there",
    );
}

#[test]
fn a_copy_chases_through_to_its_source() {
    let mut fd = build_fd();

    let leftover = unwritten(&mut fd, 0x2000, 8);
    let copied = mk_def(&mut fd, OpCode::CPUI_COPY, &[leftover], 0x2100);
    assert!(!computes_a_value(&fd, copied, 0), "a copy of leftover is still leftover");

    let k = fd.new_constant(8, 5);
    let copied_k = mk_def(&mut fd, OpCode::CPUI_COPY, &[k], 0x2200);
    assert!(computes_a_value(&fd, copied_k, 0), "a copy of a literal is a real value");
}

#[test]
fn arithmetic_is_a_computed_return_value() {
    let mut fd = build_fd();
    // Both inputs unwritten, but INT_ADD is not a move — the walk stops here.
    // This is the guard that keeps a genuine struct return whose halves are built
    // from parameters.
    let a = unwritten(&mut fd, 0x2000, 8);
    let b = unwritten(&mut fd, 0x2008, 8);
    let sum = mk_def(&mut fd, OpCode::CPUI_INT_ADD, &[a, b], 0x2100);
    assert!(computes_a_value(&fd, sum, 0), "arithmetic produces a value");
}

#[test]
fn a_load_is_a_computed_return_value() {
    let mut fd = build_fd();
    // `struct S get(struct S *p) { return *p; }` builds both halves with LOADs
    // through an (unwritten) parameter. A LOAD is not a move, so the walk stops
    // and the half is kept — this is the false positive this rule must not have.
    let spc = fd.new_constant(4, 0);
    let ptr = unwritten(&mut fd, 0x2000, 8);
    let loaded = mk_def(&mut fd, OpCode::CPUI_LOAD, &[spc, ptr], 0x2100);
    assert!(
        computes_a_value(&fd, loaded, 0),
        "a load through a pointer is a real value — a struct return reads this way",
    );
}

#[test]
fn a_phi_is_computed_when_any_arm_is() {
    let mut fd = build_fd();
    let leftover = unwritten(&mut fd, 0x2000, 8);
    let k = fd.new_constant(8, 3);
    let phi = mk_def(&mut fd, OpCode::CPUI_MULTIEQUAL, &[leftover, k], 0x2100);
    assert!(
        computes_a_value(&fd, phi, 0),
        "one real arm is enough — the function returns a value on that path",
    );
}

#[test]
fn a_phi_of_only_leftover_is_not_computed() {
    let mut fd = build_fd();
    let a = unwritten(&mut fd, 0x2000, 8);
    let b = unwritten(&mut fd, 0x2008, 8);
    let phi = mk_def(&mut fd, OpCode::CPUI_MULTIEQUAL, &[a, b], 0x2100);
    assert!(
        !computes_a_value(&fd, phi, 0),
        "leftover on every path is still leftover",
    );
}

#[test]
fn the_strict_walk_refuses_a_phi_one_arm_of_which_is_leftover() {
    let mut fd = build_fd();
    let leftover = unwritten(&mut fd, 0x2000, 8);
    let k = fd.new_constant(8, 3);
    let phi = mk_def(&mut fd, OpCode::CPUI_MULTIEQUAL, &[leftover, k], 0x2100);
    assert!(
        computes_a_value(&fd, phi, 0),
        "the pair repair's question: one real arm is enough",
    );
    assert!(
        !computes_everywhere(&fd, phi, None, false),
        "a caller adopting the whole value needs every path to produce one",
    );
}

#[test]
fn the_strict_walk_refuses_a_piece_whose_high_half_is_leftover() {
    let mut fd = build_fd();
    let leftover = unwritten(&mut fd, 0x2000, 4);
    let k = fd.new_constant(4, 7);
    let joined = mk_def(&mut fd, OpCode::CPUI_PIECE, &[leftover, k], 0x2100);
    assert!(
        !computes_everywhere(&fd, joined, None, false),
        "version_etc_arn's CONCAT44(<leftover>, call result) is not a return value",
    );
}

#[test]
fn the_strict_walk_keeps_a_value_every_byte_of_which_is_computed() {
    let mut fd = build_fd();
    let k1 = fd.new_constant(4, 1);
    let k2 = fd.new_constant(4, 2);
    let joined = mk_def(&mut fd, OpCode::CPUI_PIECE, &[k1, k2], 0x2100);
    let copy = mk_def(&mut fd, OpCode::CPUI_COPY, &[joined], 0x2200);
    assert!(
        computes_everywhere(&fd, copy, None, false),
        "a value built from constants is a return value in every byte",
    );
}

#[test]
fn the_strict_walk_stops_at_an_operation_that_produces_a_value() {
    let mut fd = build_fd();
    let leftover = unwritten(&mut fd, 0x2000, 8);
    let leftover = fd.set_input_varnode(leftover).expect("function input");
    let sum = mk_def(&mut fd, OpCode::CPUI_INT_ADD, &[leftover, leftover], 0x2100);
    assert!(
        computes_everywhere(&fd, sum, None, false),
        "arithmetic over leftover is still a value the function computed",
    );
}

// --- the pair repair on a value held in ONE return register -----------------

fn build_fd_with_blocks() -> (Funcdata, crate::context::BlockId) {
    let mut m = build_manager();
    m.insert_space(Rc::new(AddrSpace::new(
        spacetype::IPTR_PROCESSOR,
        "breg",
        true,
        8,
        1,
        3,
        0,
        1,
        1,
    )))
    .unwrap();
    let glb = Rc::new(ArchContext::new(m));
    let ram = Rc::clone(glb.manage().get_space_by_name("ram").unwrap());
    let mut fd = Funcdata::new("func", "func", glb, Address::new(ram, 0x1000), 0x1000_0000, 0x40).unwrap();
    let root = fd.bblocks_ref().root.expect("bblocks root");
    let bl = fd.bblocks_mut().new_block_basic(root);
    (fd, bl)
}

/// A live `out = <opc>(inputs...)` in `bl`, `out` sized `size` at ram `out_off`.
fn live_def(fd: &mut Funcdata, bl: crate::context::BlockId, opc: OpCode, inputs: &[VarnodeId], out_off: u64, size: int4) -> VarnodeId {
    let r = ram(fd);
    let op = fd.new_op(inputs.len() as int4, Address::new(Rc::clone(&r), out_off));
    fd.op_set_opcode(op, TypeOp::new(opc, 0, format!("{opc:?}")));
    for (i, &vn) in inputs.iter().enumerate() {
        fd.op_set_input(op, vn, i as int4).expect("wire input");
    }
    fd.op_insert(op, bl, None);
    fd.new_varnode_out(size, &Address::new(r, out_off), op).expect("varnode out")
}

/// `return <value>;` as a live RETURN in `bl`.
fn live_return(fd: &mut Funcdata, bl: crate::context::BlockId, value: VarnodeId) -> OpId {
    let r = ram(fd);
    let op = fd.new_op(2, Address::new(r, 0x1ff0));
    fd.op_set_opcode(op, TypeOp::new(OpCode::CPUI_RETURN, 0, "RETURN"));
    let k = fd.new_constant(8, 0);
    fd.op_set_input(op, k, 0).expect("wire input");
    fd.op_set_input(op, value, 1).expect("wire input");
    fd.op_insert(op, bl, None);
    op
}

#[test]
fn each_half_of_one_register_sits_in_its_own_bytes() {
    let (mut fd, _) = build_fd_with_blocks();
    let r = ram(&fd);
    let whole = fd.new_varnode(8, &Address::new(Rc::clone(&r), 0x3000), None);
    assert_eq!(slot_storage(&fd, whole, 0, 4), Some(Address::new(Rc::clone(&r), 0x3000)), "little-endian low half");
    assert_eq!(slot_storage(&fd, whole, 4, 4), Some(Address::new(Rc::clone(&r), 0x3004)), "little-endian high half");
    assert!(!spans_two_locations(&fd, whole), "one register is one location");

    let be = Rc::clone(fd.get_arch().manage().get_space_by_name("breg").unwrap());
    let whole_be = fd.new_varnode(8, &Address::new(Rc::clone(&be), 0x100), None);
    assert_eq!(slot_storage(&fd, whole_be, 0, 4), Some(Address::new(Rc::clone(&be), 0x104)), "big-endian low half");
    assert_eq!(slot_storage(&fd, whole_be, 4, 4), Some(Address::new(be, 0x100)), "big-endian high half");
}

#[test]
fn a_never_written_high_half_of_one_register_is_still_dropped() {
    // `RAX = PIECE(<RAX's own high bytes, never written>, EAX = a + b)`: the
    // function computed only the low half into its return register.
    let (mut fd, bl) = build_fd_with_blocks();
    let r = ram(&fd);
    let leftover = fd.new_varnode(4, &Address::new(Rc::clone(&r), 0x3004), None);
    let a = unwritten(&mut fd, 0x2000, 4);
    let b = unwritten(&mut fd, 0x2008, 4);
    let sum = live_def(&mut fd, bl, OpCode::CPUI_INT_ADD, &[a, b], 0x3000, 4);
    let whole = live_def(&mut fd, bl, OpCode::CPUI_PIECE, &[leftover, sum], 0x3000, 8);
    let ret = live_return(&mut fd, bl, whole);
    assert!(strip_uncomputed_return_piece(&mut fd));
    assert_eq!(fd.obank().get(ret).unwrap().get_in(1), Some(sum), "the return narrows to the computed low half");
}

#[test]
fn the_high_half_of_one_register_is_never_returned_alone() {
    // `RAX = PIECE(a + b, <EAX, never written>)`: returning the sum by itself
    // would hand back the high 32 bits as the whole value.
    let (mut fd, bl) = build_fd_with_blocks();
    let r = ram(&fd);
    let a = unwritten(&mut fd, 0x2000, 4);
    let b = unwritten(&mut fd, 0x2008, 4);
    let sum = live_def(&mut fd, bl, OpCode::CPUI_INT_ADD, &[a, b], 0x3100, 4);
    let leftover = fd.new_varnode(4, &Address::new(Rc::clone(&r), 0x3000), None);
    let whole = live_def(&mut fd, bl, OpCode::CPUI_PIECE, &[sum, leftover], 0x3000, 8);
    let ret = live_return(&mut fd, bl, whole);
    assert!(!strip_uncomputed_return_piece(&mut fd), "nothing is dropped from a value built in one register");
    assert_eq!(fd.obank().get(ret).unwrap().get_in(1), Some(whole), "the return keeps all eight bytes");
}

/// A live op with no output in `bl`: a store, a branch, a call.
fn live_sink(fd: &mut Funcdata, bl: crate::context::BlockId, opc: OpCode, inputs: &[VarnodeId]) -> OpId {
    let r = ram(fd);
    let op = fd.new_op(inputs.len() as int4, Address::new(r, 0x1f00));
    fd.op_set_opcode(op, TypeOp::new(opc, 0, format!("{opc:?}")));
    for (i, &vn) in inputs.iter().enumerate() {
        fd.op_set_input(op, vn, i as int4).expect("wire input");
    }
    fd.op_insert(op, bl, None);
    op
}

/// `%o1 = <window copy of> value; return %o0, %o1;` with `first` in `%o0`:
/// the restored register and the copy that restores it.
fn restore_and_return_with(
    fd: &mut Funcdata,
    bl: crate::context::BlockId,
    first: VarnodeId,
    value: VarnodeId,
) -> (VarnodeId, OpId) {
    let restored = live_def(fd, bl, OpCode::CPUI_COPY, &[value], 0x2000, 4);
    let copy = fd.vbank().get(restored).and_then(|v| v.get_def()).expect("restore copy");
    let r = ram(fd);
    let op = fd.new_op(3, Address::new(r, 0x1ff0));
    fd.op_set_opcode(op, TypeOp::new(OpCode::CPUI_RETURN, 0, "RETURN"));
    let k = fd.new_constant(8, 0);
    fd.op_set_input(op, k, 0).expect("wire input");
    fd.op_set_input(op, first, 1).expect("wire input");
    fd.op_set_input(op, restored, 2).expect("wire input");
    fd.op_insert(op, bl, None);
    (restored, copy)
}

/// [`restore_and_return_with`] an unrelated value in `%o0`.
fn restore_and_return(fd: &mut Funcdata, bl: crate::context::BlockId, value: VarnodeId) -> (VarnodeId, OpId) {
    let first = fd.new_varnode(4, &Address::new(ram(fd), 0x2800), None);
    restore_and_return_with(fd, bl, first, value)
}

/// What the window hands back in `restored` (slot 2 of its RETURN, `%o0` in
/// slot 1), the copies in `windows` counting as window moves.
fn handed(fd: &Funcdata, restored: VarnodeId, windows: &[OpId]) -> HandedValue {
    let o1 = Address::new(ram(fd), 0x2000);
    window_values(fd, &[restored], &o1, 1, &mut |op| windows.contains(&op))[0]
}

/// SPARC's `restore` copies `%i1` back to `%o1`, and `save` copied `%o1` into
/// `%i1`: the returned `%o1` is the second argument passed straight back, a
/// literal or a value the function left in `%i1`. One it put there only to be
/// returned is deliberate; one it also stored, branched on, called with or
/// loaded through was left there by that other use.
#[test]
fn a_window_hands_back_a_leftover_a_used_value_or_a_deliberate_one() {
    let (mut fd, bl) = build_fd_with_blocks();
    let entry = fd.new_varnode(4, &Address::new(ram(&fd), 0x2000), None);
    let entry = fd.set_input_varnode(entry).expect("function input");
    let saved = live_def(&mut fd, bl, OpCode::CPUI_COPY, &[entry], 0x2100, 4);
    let save = fd.vbank().get(saved).and_then(|v| v.get_def()).unwrap();
    let k = fd.new_constant(4, 1);
    let sum = live_def(&mut fd, bl, OpCode::CPUI_INT_ADD, &[entry, k], 0x2400, 4);
    let at = fd.new_constant(8, 0x9000);
    let spc = fd.new_constant(8, 0);
    live_sink(&mut fd, bl, OpCode::CPUI_STORE, &[spc, at, sum]);
    let (restored, restore) = restore_and_return(&mut fd, bl, saved);
    assert_eq!(handed(&fd, restored, &[save, restore]), HandedValue::Leftover, "the entry value, although the function reads it");
    assert_eq!(
        handed(&fd, restored, &[restore]),
        HandedValue::Deliberate,
        "a copy that is not a window's is a move the function made to return the value",
    );

    let (mut fd, bl) = build_fd_with_blocks();
    let zero = fd.new_constant(4, 0);
    let scratch = live_def(&mut fd, bl, OpCode::CPUI_COPY, &[zero], 0x2100, 4);
    let (at, spc) = (fd.new_constant(8, 0x9000), fd.new_constant(8, 0));
    live_sink(&mut fd, bl, OpCode::CPUI_STORE, &[spc, at, scratch]);
    let (restored, restore) = restore_and_return(&mut fd, bl, scratch);
    assert_eq!(handed(&fd, restored, &[restore]), HandedValue::Leftover, "a zero the function stored first");

    let (mut fd, bl) = build_fd_with_blocks();
    let ten = fd.new_constant(4, 10);
    let low = live_def(&mut fd, bl, OpCode::CPUI_COPY, &[ten], 0x2100, 4);
    let (restored, restore) = restore_and_return(&mut fd, bl, low);
    assert_eq!(handed(&fd, restored, &[restore]), HandedValue::Deliberate, "`mov 10,%i1; ret; restore`");

    let (mut fd, bl) = build_fd_with_blocks();
    let minus_one = fd.new_constant(4, 0xffff_ffff);
    let low = live_def(&mut fd, bl, OpCode::CPUI_COPY, &[minus_one], 0x2100, 4);
    let z = fd.new_constant(4, 0);
    let high = live_def(&mut fd, bl, OpCode::CPUI_INT_ADD, &[z, low], 0x2300, 4);
    let (restored, restore) = restore_and_return_with(&mut fd, bl, high, low);
    assert_eq!(
        handed(&fd, restored, &[restore]),
        HandedValue::Deliberate,
        "`mov -1,%i1; ret; restore %g0,%i1,%o0` returns -1 in both registers, a `long long` -1",
    );

    let (mut fd, bl) = build_fd_with_blocks();
    let two = fd.new_constant(4, 2);
    let low = live_def(&mut fd, bl, OpCode::CPUI_COPY, &[two], 0x2100, 4);
    let z = fd.new_constant(4, 0);
    let high = live_def(&mut fd, bl, OpCode::CPUI_INT_ADD, &[z, low], 0x2300, 4);
    let (restored, restore) = restore_and_return_with(&mut fd, bl, high, low);
    assert_eq!(
        handed(&fd, restored, &[restore]),
        HandedValue::Leftover,
        "`mov 2,%i1; ret; restore %g0,%i1,%o0` builds an int 2 in %i1; 0x200000002 is no value",
    );

    let (mut fd, bl) = build_fd_with_blocks();
    let spc = fd.new_constant(8, 0);
    let at = fd.new_varnode(4, &Address::new(ram(&fd), 0x3000), None);
    let at = fd.set_input_varnode(at).expect("function input");
    let reload = live_def(&mut fd, bl, OpCode::CPUI_LOAD, &[spc, at], 0x2100, 4);
    let z = fd.new_constant(4, 0);
    let high = live_def(&mut fd, bl, OpCode::CPUI_INT_ADD, &[z, reload], 0x2300, 4);
    let (restored, restore) = restore_and_return_with(&mut fd, bl, high, reload);
    assert_eq!(
        handed(&fd, restored, &[restore]),
        HandedValue::Deliberate,
        "clang -O0's `ld [%fp-8],%i1; ret; restore %g0,%i1,%o0` for `c ? -1 : 0` reads right as either type",
    );

    let (mut fd, bl) = build_fd_with_blocks();
    let one = fd.new_constant(4, 1);
    let flag = live_def(&mut fd, bl, OpCode::CPUI_COPY, &[one], 0x2100, 4);
    let x = fd.new_varnode(4, &Address::new(ram(&fd), 0x3000), None);
    let x = fd.set_input_varnode(x).expect("function input");
    let high = live_def(&mut fd, bl, OpCode::CPUI_INT_OR, &[flag, x], 0x2300, 4);
    let (restored, restore) = restore_and_return_with(&mut fd, bl, high, flag);
    assert_eq!(
        handed(&fd, restored, &[restore]),
        HandedValue::Leftover,
        "`or %i1,%i0,%i0` ORs a flag left in %i1 into the int in %o0",
    );

    let (mut fd, bl) = build_fd_with_blocks();
    let spc = fd.new_constant(8, 0);
    let at = fd.new_varnode(4, &Address::new(ram(&fd), 0x3000), None);
    let at = fd.set_input_varnode(at).expect("function input");
    let byte = live_def(&mut fd, bl, OpCode::CPUI_LOAD, &[spc, at], 0x2500, 1);
    let widened = live_def(&mut fd, bl, OpCode::CPUI_INT_ZEXT, &[byte], 0x2100, 4);
    let z = fd.new_constant(4, 0);
    let high = live_def(&mut fd, bl, OpCode::CPUI_INT_ADD, &[z, widened], 0x2300, 4);
    let (restored, restore) = restore_and_return_with(&mut fd, bl, high, widened);
    assert_eq!(
        handed(&fd, restored, &[restore]),
        HandedValue::Leftover,
        "`ldub [..],%i1; ret; restore %g0,%i1,%o0` returns the byte as an int: its top bit is clear, so the pair is no sign extension",
    );

    let (mut fd, bl) = build_fd_with_blocks();
    let five = fd.new_constant(4, 5);
    let low = live_def(&mut fd, bl, OpCode::CPUI_COPY, &[five], 0x2100, 4);
    let k31 = fd.new_constant(4, 31);
    let high = live_def(&mut fd, bl, OpCode::CPUI_INT_SRIGHT, &[low, k31], 0x2300, 4);
    let (restored, restore) = restore_and_return_with(&mut fd, bl, high, low);
    assert_eq!(handed(&fd, restored, &[restore]), HandedValue::Deliberate, "`mov 5,%i1; sra %i1,31,%i0` is the long long 5");

    let (mut fd, bl) = build_fd_with_blocks();
    let zero = fd.new_constant(4, 0);
    let nought = live_def(&mut fd, bl, OpCode::CPUI_COPY, &[zero], 0x2100, 4);
    let x = fd.new_varnode(4, &Address::new(ram(&fd), 0x3000), None);
    let x = fd.set_input_varnode(x).expect("function input");
    let three = fd.new_constant(4, 3);
    let product = live_def(&mut fd, bl, OpCode::CPUI_INT_MULT, &[x, three], 0x2200, 4);
    let product_high = live_def(&mut fd, bl, OpCode::CPUI_INT_RIGHT, &[product, three], 0x2400, 4);
    let low = live_def(&mut fd, bl, OpCode::CPUI_MULTIEQUAL, &[nought, product], 0x2100, 4);
    let high = live_def(&mut fd, bl, OpCode::CPUI_MULTIEQUAL, &[nought, product_high], 0x2300, 4);
    let (restored, restore) = restore_and_return_with(&mut fd, bl, high, low);
    assert_eq!(
        handed(&fd, restored, &[restore]),
        HandedValue::Deliberate,
        "a (0, 0) path beside a computed pair keeps the pair",
    );

    let (mut fd, bl) = build_fd_with_blocks();
    let spc = fd.new_constant(8, 0);
    let at = fd.new_varnode(4, &Address::new(ram(&fd), 0x3000), None);
    let at = fd.set_input_varnode(at).expect("function input");
    let low = live_def(&mut fd, bl, OpCode::CPUI_LOAD, &[spc, at], 0x2100, 4);
    let k31 = fd.new_constant(4, 31);
    let high = live_def(&mut fd, bl, OpCode::CPUI_INT_SRIGHT, &[low, k31], 0x2300, 4);
    let (restored, restore) = restore_and_return_with(&mut fd, bl, high, low);
    assert_eq!(
        handed(&fd, restored, &[restore]),
        HandedValue::Deliberate,
        "`sra %i1,31,%i0` makes %o0 the sign of %i1: a sign-extended long long",
    );

    let (mut fd, bl) = build_fd_with_blocks();
    let page = fd.new_constant(4, 0x9000);
    let base = live_def(&mut fd, bl, OpCode::CPUI_COPY, &[page], 0x2100, 4);
    let spc = fd.new_constant(8, 0);
    let loaded = live_def(&mut fd, bl, OpCode::CPUI_LOAD, &[spc, base], 0x2300, 4);
    live_return(&mut fd, bl, loaded);
    let (restored, restore) = restore_and_return(&mut fd, bl, base);
    assert_eq!(handed(&fd, restored, &[restore]), HandedValue::Leftover, "an address half `sethi` built to load through");

    let (mut fd, bl) = build_fd_with_blocks();
    let x = fd.new_varnode(4, &Address::new(ram(&fd), 0x3000), None);
    let x = fd.set_input_varnode(x).expect("function input");
    let doubled = live_def(&mut fd, bl, OpCode::CPUI_INT_ADD, &[x, x], 0x2100, 4);
    let z = fd.new_constant(4, 0);
    let test = live_def(&mut fd, bl, OpCode::CPUI_INT_EQUAL, &[doubled, z], 0x2500, 1);
    let dest = fd.new_constant(8, 0x6000);
    live_sink(&mut fd, bl, OpCode::CPUI_CBRANCH, &[dest, test]);
    let (restored, restore) = restore_and_return(&mut fd, bl, doubled);
    assert_eq!(handed(&fd, restored, &[restore]), HandedValue::Computed, "a value the function branched on");

    let (mut fd, bl) = build_fd_with_blocks();
    let x = fd.new_varnode(4, &Address::new(ram(&fd), 0x3000), None);
    let x = fd.set_input_varnode(x).expect("function input");
    let doubled = live_def(&mut fd, bl, OpCode::CPUI_INT_ADD, &[x, x], 0x2100, 4);
    let (restored, restore) = restore_and_return(&mut fd, bl, doubled);
    assert_eq!(handed(&fd, restored, &[restore]), HandedValue::Deliberate, "a low word computed to be returned");

    let (mut fd, bl) = build_fd_with_blocks();
    let zero = fd.new_constant(4, 0);
    let whole = live_def(&mut fd, bl, OpCode::CPUI_COPY, &[zero], 0x2100, 4);
    let (b0, b1) = (fd.new_constant(4, 0), fd.new_constant(4, 1));
    let low_byte = live_def(&mut fd, bl, OpCode::CPUI_SUBPIECE, &[whole, b0], 0x2103, 1);
    let high_bytes = live_def(&mut fd, bl, OpCode::CPUI_SUBPIECE, &[whole, b1], 0x2100, 3);
    let joined = live_def(&mut fd, bl, OpCode::CPUI_PIECE, &[high_bytes, low_byte], 0x2100, 4);
    let (at, spc) = (fd.new_constant(8, 0x9000), fd.new_constant(8, 0));
    live_sink(&mut fd, bl, OpCode::CPUI_STORE, &[spc, at, low_byte]);
    let (restored, restore) = restore_and_return(&mut fd, bl, joined);
    assert_eq!(
        handed(&fd, restored, &[restore]),
        HandedValue::Leftover,
        "`mov %g0,%i1; stb %i1,[..]` stores the low byte heritage split off",
    );

    let (mut fd, bl) = build_fd_with_blocks();
    let entry = fd.new_varnode(4, &Address::new(ram(&fd), 0x2000), None);
    let entry = fd.set_input_varnode(entry).expect("function input");
    let ten = fd.new_constant(4, 10);
    let low = live_def(&mut fd, bl, OpCode::CPUI_COPY, &[ten], 0x2100, 4);
    let merged = live_def(&mut fd, bl, OpCode::CPUI_MULTIEQUAL, &[entry, low], 0x2100, 4);
    let (restored, restore) = restore_and_return(&mut fd, bl, merged);
    assert_eq!(handed(&fd, restored, &[restore]), HandedValue::Deliberate, "one path puts a literal there to return it");
    let (mut fd, bl) = build_fd_with_blocks();
    let entry = fd.new_varnode(4, &Address::new(ram(&fd), 0x2000), None);
    let entry = fd.set_input_varnode(entry).expect("function input");
    let merged = live_def(&mut fd, bl, OpCode::CPUI_MULTIEQUAL, &[entry, entry], 0x2100, 4);
    let (restored, restore) = restore_and_return(&mut fd, bl, merged);
    assert_eq!(handed(&fd, restored, &[restore]), HandedValue::Leftover, "every path hands the entry value back");
}

/// `out = <opc>(inputs...)`, `out` sized `size` at ram `out_off`.
fn mk_sized(fd: &mut Funcdata, opc: OpCode, inputs: &[VarnodeId], out_off: u64, size: int4) -> VarnodeId {
    let r = ram(fd);
    let op = fd.new_op(inputs.len() as int4, Address::new(Rc::clone(&r), out_off));
    fd.obank_mut().change_opcode(op, TypeOp::new(opc, 0, format!("{opc:?}")));
    for (i, &vn) in inputs.iter().enumerate() {
        fd.op_set_input(op, vn, i as int4).expect("wire input");
    }
    fd.new_varnode_out(size, &Address::new(r, out_off), op).expect("varnode out")
}

/// SPARC builds a constant in two instructions (`sethi 0,%i1; add %i1,0,%i1`),
/// and the early drop meets them before the rule pool folds them.
#[test]
fn a_literal_may_be_built_in_steps() {
    let mut fd = build_fd();
    let (z0, z1) = (fd.new_constant(4, 0), fd.new_constant(4, 0));
    let hi = mk_sized(&mut fd, OpCode::CPUI_COPY, &[z0], 0x2000, 4);
    let whole = mk_sized(&mut fd, OpCode::CPUI_INT_ADD, &[hi, z1], 0x2004, 4);
    assert!(built_from_literals(&fd, whole, LITERAL_DEPTH), "sethi then add of literals is a literal");

    let x = unwritten(&mut fd, 0x3000, 4);
    let one = fd.new_constant(4, 1);
    let sum = mk_sized(&mut fd, OpCode::CPUI_INT_ADD, &[x, one], 0x2008, 4);
    assert!(!built_from_literals(&fd, sum, LITERAL_DEPTH), "arithmetic on an input is a value");
    let space = fd.new_constant(8, 3);
    let load = mk_sized(&mut fd, OpCode::CPUI_LOAD, &[space, hi], 0x200c, 4);
    assert!(!built_from_literals(&fd, load, LITERAL_DEPTH), "a load through a literal address reads memory");
}

/// The late narrowing asks whether the low word of a returned pair is zero
/// whatever the inputs hold.
#[test]
fn the_low_word_is_zero_by_shift_mask_or_concatenation() {
    let mut fd = build_fd();
    let k = fd.new_constant(8, 0x4000_0002_0000_0000);
    assert!(low_bits_zero(&fd, k, 32, ZERO_DEPTH), "a literal whose low word is zero");
    let four = fd.new_constant(8, 4);
    assert!(!low_bits_zero(&fd, four, 32, ZERO_DEPTH), "`return 4` as a pair has a low word");

    let x = unwritten(&mut fd, 0x3000, 4);
    let x = fd.set_input_varnode(x).expect("function input");
    let wide = mk_sized(&mut fd, OpCode::CPUI_INT_ZEXT, &[x], 0x2000, 8);
    let by32 = fd.new_constant(4, 32);
    let shifted = mk_sized(&mut fd, OpCode::CPUI_INT_LEFT, &[wide, by32], 0x2008, 8);
    assert!(low_bits_zero(&fd, shifted, 32, ZERO_DEPTH), "ZEXT(x) << 32, what PIECE(x, 0) folds into");
    assert!(!low_bits_zero(&fd, wide, 32, ZERO_DEPTH), "ZEXT(x) holds x in the low word");
    let mask = fd.new_constant(8, 0xffff_ffff_0000_0000);
    let flipped = mk_sized(&mut fd, OpCode::CPUI_INT_XOR, &[shifted, mask], 0x2010, 8);
    assert!(low_bits_zero(&fd, flipped, 32, ZERO_DEPTH), "flipping the high word keeps the low one zero");
    let y = unwritten(&mut fd, 0x3100, 8);
    let y = fd.set_input_varnode(y).expect("function input");
    let ored = mk_sized(&mut fd, OpCode::CPUI_INT_OR, &[shifted, y], 0x2018, 8);
    assert!(!low_bits_zero(&fd, ored, 32, ZERO_DEPTH), "or-ing in an unknown value can set the low word");

    let zero4 = fd.new_constant(4, 0);
    let low_zero = mk_sized(&mut fd, OpCode::CPUI_PIECE, &[x, zero4], 0x2020, 8);
    assert!(low_bits_zero(&fd, low_zero, 32, ZERO_DEPTH), "PIECE(x, 0)");
    let zero4b = fd.new_constant(4, 0);
    let high_zero = mk_sized(&mut fd, OpCode::CPUI_PIECE, &[zero4b, x], 0x2028, 8);
    assert!(!low_bits_zero(&fd, high_zero, 32, ZERO_DEPTH), "PIECE(0, x) holds x in the low word");
    let phi = mk_sized(&mut fd, OpCode::CPUI_MULTIEQUAL, &[shifted, low_zero], 0x2030, 8);
    assert!(low_bits_zero(&fd, phi, 32, ZERO_DEPTH), "every path has a zero low word");
    let mixed = mk_sized(&mut fd, OpCode::CPUI_MULTIEQUAL, &[shifted, high_zero], 0x2038, 8);
    assert!(!low_bits_zero(&fd, mixed, 32, ZERO_DEPTH), "one path returns x in the low word");
}

/// The late repair finds the first register's value in the PIECE, in the
/// zero-extension of a PIECE over the second register, in `ZEXT(x) << 32` and
/// in a literal pair; any other fold is narrowed before it.
#[test]
fn the_late_repair_reads_the_join_and_its_two_folds() {
    let mut fd = build_fd();
    let x = unwritten(&mut fd, 0x3000, 4);
    let x = fd.set_input_varnode(x).expect("function input");
    let y = unwritten(&mut fd, 0x3008, 4);
    let y = fd.set_input_varnode(y).expect("function input");
    let piece = mk_sized(&mut fd, OpCode::CPUI_PIECE, &[x, y], 0x2000, 8);
    assert!(late_strippable(&fd, piece, 4));
    let b = unwritten(&mut fd, 0x3100, 1);
    let b = fd.set_input_varnode(b).expect("function input");
    let five = mk_sized(&mut fd, OpCode::CPUI_PIECE, &[b, y], 0x2008, 5);
    let bool_high = mk_sized(&mut fd, OpCode::CPUI_INT_ZEXT, &[five], 0x2010, 8);
    assert!(late_strippable(&fd, bool_high, 4), "ZEXT(PIECE(b, y))");
    let wide = mk_sized(&mut fd, OpCode::CPUI_INT_ZEXT, &[x], 0x2018, 8);
    let by32 = fd.new_constant(4, 32);
    let shifted = mk_sized(&mut fd, OpCode::CPUI_INT_LEFT, &[wide, by32], 0x2020, 8);
    assert!(late_strippable(&fd, shifted, 4), "ZEXT(x) << 32");
    assert_eq!(shifted_high_half(&fd, shifted), Some(x));
    assert!(!late_strippable(&fd, wide, 4), "ZEXT(y) lost the high half");
    let mask = fd.new_constant(8, 0xffff_ffff_0000_0000);
    let flipped = mk_sized(&mut fd, OpCode::CPUI_INT_XOR, &[shifted, mask], 0x2028, 8);
    assert!(!late_strippable(&fd, flipped, 4), "no Varnode holds x ^ 0xffffffff");
    let k = fd.new_constant(8, 0x4000_0002_0000_0000);
    let literal = mk_sized(&mut fd, OpCode::CPUI_COPY, &[k], 0x2030, 8);
    assert!(late_strippable(&fd, literal, 4), "PIECE(#k, #0) folds into one literal");
    assert_eq!(literal_pair(&fd, literal), Some((0x4000_0002_0000_0000, 4, 4)));
}

/// A condition built from literals by copies, `==`, `!=` and `!` is decided;
/// anything a call or an input feeds is not.
#[test]
fn a_condition_of_literals_is_decided() {
    let mut fd = build_fd();
    let z = fd.new_constant(1, 0);
    let flag = mk_sized(&mut fd, OpCode::CPUI_COPY, &[z], 0x5010, 1);
    let z2 = fd.new_constant(1, 0);
    let eq = mk_sized(&mut fd, OpCode::CPUI_INT_EQUAL, &[flag, z2], 0x2000, 1);
    assert_eq!(decided(&fd, eq, 8), Some(1), "didrestore == 0 with didrestore = 0");
    let z3 = fd.new_constant(1, 0);
    let ne = mk_sized(&mut fd, OpCode::CPUI_INT_NOTEQUAL, &[flag, z3], 0x2001, 1);
    assert_eq!(decided(&fd, ne, 8), Some(0));
    let not = mk_sized(&mut fd, OpCode::CPUI_BOOL_NEGATE, &[ne], 0x2002, 1);
    assert_eq!(decided(&fd, not, 8), Some(1));
    let input = unwritten(&mut fd, 0x5020, 1);
    let input = fd.set_input_varnode(input).expect("function input");
    let z4 = fd.new_constant(1, 0);
    let open = mk_sized(&mut fd, OpCode::CPUI_INT_EQUAL, &[input, z4], 0x2003, 1);
    assert_eq!(decided(&fd, open, 8), None, "an input is not a literal");
}

/// SPARC's `call` pcode keeps `if (didrestore == 0) goto next; return [o7];`
/// for a `restore` in its delay slot. With `didrestore = 0` the RETURN on the
/// fall-through is never reached; the edge a literal condition takes is.
#[test]
fn a_return_behind_a_decided_branch_is_never_reached() {
    let (mut fd, head) = build_fd_with_blocks();
    let root = fd.bblocks_ref().root.expect("bblocks root");
    let fall = fd.bblocks_mut().new_block_basic(root);
    let taken = fd.bblocks_mut().new_block_basic(root);
    fd.bblocks_mut().add_edge(head, fall);
    fd.bblocks_mut().add_edge(head, taken);
    let z = fd.new_constant(1, 0);
    let flag = live_def(&mut fd, head, OpCode::CPUI_COPY, &[z], 0x5010, 1);
    let z2 = fd.new_constant(1, 0);
    let cond = live_def(&mut fd, head, OpCode::CPUI_INT_EQUAL, &[flag, z2], 0x5100, 1);
    let r = ram(&fd);
    let cbranch = fd.new_op(2, Address::new(Rc::clone(&r), 0x5200));
    fd.op_set_opcode(cbranch, TypeOp::new(OpCode::CPUI_CBRANCH, 0, "CBRANCH"));
    let dest = fd.new_constant(8, 0x6000);
    fd.op_set_input(cbranch, dest, 0).expect("wire input");
    fd.op_set_input(cbranch, cond, 1).expect("wire input");
    fd.op_insert(cbranch, head, None);
    let v1 = unwritten(&mut fd, 0x3000, 8);
    let on_fall = live_return(&mut fd, fall, v1);
    let v2 = unwritten(&mut fd, 0x3008, 8);
    let on_taken = live_return(&mut fd, taken, v2);
    assert!(never_reached(&fd, on_fall), "the condition is true, so the fall-through is dead");
    assert!(!never_reached(&fd, on_taken), "the taken edge is live");
    let (mut fd2, lone) = build_fd_with_blocks();
    let v3 = unwritten(&mut fd2, 0x3000, 8);
    let entry_return = live_return(&mut fd2, lone, v3);
    assert!(!never_reached(&fd2, entry_return), "a block nothing branches to is the entry");
}
