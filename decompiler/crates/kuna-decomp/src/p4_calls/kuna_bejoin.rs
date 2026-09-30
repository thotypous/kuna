//! (kuna) The order a two-register return value is joined in on an ABI that
//! puts its HIGH word in the first register.
//!
//! PowerPC (r3:r4), MIPS o32 big-endian (v0:v1), SPARC (o0:o1), ARM big-endian
//! (r0:r1) and AVR's gcc ABI (R25:R24) return a value twice a register's width
//! with its most significant half in the first return register, and their
//! output rule says so (`ParamActive::join_pair_order`). Return recovery used
//! to join every pair first register low, so a `long long` came back with its
//! halves swapped.
//!
//! Joining in the ABI's order is only right when the second register really is
//! the low word. A function returning one register often leaves something else
//! in the second: an argument the window hands back (SPARC's `restore` copies
//! `%i1` into `%o1`), a scratch value it also used, a literal `-O0` code wrote
//! and never read (clang's `addiu $3,$zero,0` on MIPS). Joined first register
//! low, such a pair narrows back to the first register everywhere a later pass
//! drops the phantom half (the uncomputed-half repair, a boolean or byte
//! return, C's own truncation in a narrow prototype), which is the right
//! value. Joined in the ABI's order, every one of those keeps the wrong
//! register.
//!
//! So a pair is joined in the ABI's order only when its low word is returned on
//! purpose, and first register low otherwise, exactly as before. The second
//! register is judged at every live RETURN by walking back through moves to
//! where its value was made:
//!
//! * the two halves of one wide value (an 8-byte load, a call's two output
//!   registers): the function hands back a wide value, joined in the ABI's
//!   order whatever else it does -- the order a call's own pair is built in;
//! * the register's own entry value, reached without an instruction that moves
//!   it (untouched, or carried by a register window): a leftover, and the whole
//!   function keeps the old join;
//! * a value that is also used for anything but the returned pair (a store, a
//!   call, a branch, an address, the first register's value other than through
//!   its sign or a carry), or the first register's own value: scratch, and the
//!   whole function keeps the old join;
//! * a literal zero: `-O0` leaves one behind, so it proves nothing either way;
//! * anything else -- a value or a nonzero literal nothing but the RETURNs read
//!   -- is returned on purpose.
//!
//! The pair is joined in the ABI's order when some RETURN returns its low word
//! on purpose and none has a leftover or scratch there. The rule is a prior, not
//! a proof: `(u64)x << 32` leaves the same zero in the second register as a
//! function returning `int`, and a `long long` whose low word also feeds a call
//! looks like scratch; both keep the old join, which is what they printed
//! before.

use std::collections::BTreeSet;
use std::rc::Rc;

use kuna_base::address::Address;
use kuna_base::space::spacetype;
use kuna_base::types::int4;
use kuna_num::opcodes::OpCode;

use crate::context::{OpId, VarnodeId};
use crate::fspec::ParamActive;
use crate::funcdata::Funcdata;

/// How many Varnodes one walk visits before it gives up (and calls the value
/// scratch, the old join).
const MAX_NODES: usize = 4096;

/// How many operations deep [`literal_value`] looks: SPARC builds a
/// 32-bit constant in two (`sethi` then `or`).
const LITERAL_DEPTH: u32 = 4;

/// The trial indexes `(low, high)` to join the two used output trials of
/// `active` in: the ABI's order when [`low_word_returned`], first trial low
/// otherwise.
pub fn join_order(active: &ParamActive, data: &Funcdata, return_ops: &[OpId]) -> (int4, int4) {
    let abi = active.join_pair_order();
    if abi == (0, 1) || !low_word_returned(active, data, return_ops) {
        (0, 1)
    } else {
        abi
    }
}

/// Did `join_order` pick `order` against the ABI: a pair of used trials the
/// ABI joins first register high, joined first register low? The function's
/// calls then join their own pairs the same way, so a pair a call hands back
/// and the function returns stays one value.
pub fn joins_first_low(active: &ParamActive, order: (int4, int4)) -> bool {
    let used = (0..active.get_num_trials()).take_while(|&i| active.get_trial(i).is_used()).count();
    used == 2 && order == (0, 1) && active.join_pair_order() != (0, 1)
}

/// What one RETURN holds in the second register ([`classify`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum LowWord {
    /// The low half of the wide value the first register holds the high half of.
    Wide,
    /// A value or nonzero literal the function returns on purpose.
    Returned,
    /// A literal zero.
    Zero,
    /// A callee's clobber or a location the function never wrote.
    Nothing,
    /// The register's own entry value, left in place or carried by a window.
    Entry,
    /// A value the function also used for something else.
    Scratch,
}

/// Is the low word of the pair `active` joins returned on purpose at the live
/// RETURNs of `data` (see the module header)?
fn low_word_returned(active: &ParamActive, data: &Funcdata, return_ops: &[OpId]) -> bool {
    let used = (0..active.get_num_trials()).take_while(|&i| active.get_trial(i).is_used()).count();
    if used != 2 {
        return false;
    }
    let (lo, hi) = active.join_pair_order();
    let (lo_t, hi_t) = (active.get_trial(lo), active.get_trial(hi));
    let pair = Pair {
        lo_slot: lo_t.get_slot(),
        hi_slot: hi_t.get_slot(),
        lo_size: lo_t.get_size(),
        own: lo_t.get_address().clone(),
    };
    let mut returned = false;
    let mut vetoed = false;
    for &retop in return_ops {
        let Some(o) = data.obank().get(retop) else { continue };
        if o.is_dead() || o.get_halt_type() != 0 || never_reached(data, retop) {
            continue;
        }
        match classify(data, retop, &pair) {
            LowWord::Wide => return true,
            LowWord::Returned => returned = true,
            LowWord::Entry | LowWord::Scratch => vetoed = true,
            LowWord::Zero | LowWord::Nothing => {}
        }
    }
    returned && !vetoed
}

/// Where the two trials sit in a RETURN, and the low word's register.
struct Pair {
    lo_slot: int4,
    hi_slot: int4,
    lo_size: int4,
    own: Address,
}

/// Classify the second register `retop` returns.
fn classify(data: &Funcdata, retop: OpId, pair: &Pair) -> LowWord {
    let Some(o) = data.obank().get(retop) else { return LowWord::Scratch };
    let (Some(low), Some(high)) = (o.get_in(pair.lo_slot), o.get_in(pair.hi_slot)) else {
        return LowWord::Scratch;
    };
    if halves_of_one_value(data, low, high, pair.lo_size) {
        return LowWord::Wide;
    }
    let ends = ends_of(data, low, &pair.own, pair.lo_size);
    if ends.overflow {
        return LowWord::Scratch;
    }
    if ends.values.iter().any(|&v| !only_returned(data, v, pair)) {
        return LowWord::Scratch;
    }
    if ends.entry {
        return LowWord::Entry;
    }
    if ends.values.is_empty() {
        return LowWord::Nothing;
    }
    if ends.values.iter().all(|&v| literal_value(data, v, LITERAL_DEPTH) == Some(0)) {
        return LowWord::Zero;
    }
    LowWord::Returned
}

/// Are `low` and `high`, through copies, the halves of one value at least two
/// registers wide: `SUBPIECE(w, 0)` and `SUBPIECE(w, lo_size)`?
fn halves_of_one_value(data: &Funcdata, low: VarnodeId, high: VarnodeId, lo_size: int4) -> bool {
    let piece_of = |vn: VarnodeId| -> Option<(VarnodeId, u64)> {
        let mut cur = vn;
        for _ in 0..8 {
            let op = data.obank().get(data.vbank().get(cur)?.get_def()?)?;
            match op.code() {
                OpCode::CPUI_COPY => cur = op.get_in(0)?,
                OpCode::CPUI_INDIRECT if !op.is_indirect_creation() => cur = op.get_in(0)?,
                OpCode::CPUI_SUBPIECE => {
                    let at = data.vbank().get(op.get_in(1)?)?;
                    return at.is_constant().then(|| (op.get_in(0), at.get_offset())).and_then(|(w, k)| Some((w?, k)));
                }
                _ => return None,
            }
        }
        None
    };
    match (piece_of(low), piece_of(high)) {
        (Some((w, 0)), Some((w2, k))) => w == w2 && k == lo_size as u64,
        _ => false,
    }
}

/// Where a returned value was made ([`ends_of`]).
#[derive(Default)]
struct Ends {
    /// Values and literals the walk stopped at.
    values: Vec<VarnodeId>,
    /// The register's own entry value, reached with no instruction moving it.
    entry: bool,
    /// The walk ran out of budget.
    overflow: bool,
}

/// Walk back from `low` through moves -- copies, phis, indirects, a register
/// heritage split into pieces -- to where its value was made. A copy that a
/// register window makes ([`moves_register_window`]) does not count as the
/// function moving the value; any other does, so the entry value of `own`
/// reached through one is a value the function put back on purpose (ARM's
/// `mov r4,r1; bl ext; mov r1,r4`).
fn ends_of(data: &Funcdata, low: VarnodeId, own: &Address, own_size: int4) -> Ends {
    let mut ends = Ends::default();
    let mut seen: BTreeSet<(VarnodeId, bool)> = BTreeSet::new();
    let mut work = vec![(low, false)];
    while let Some((cur, moved)) = work.pop() {
        if !seen.insert((cur, moved)) {
            continue;
        }
        if seen.len() > MAX_NODES {
            ends.overflow = true;
            break;
        }
        let Some(v) = data.vbank().get(cur) else { continue };
        if v.is_constant() {
            ends.values.push(cur);
            continue;
        }
        let Some(def) = v.get_def() else {
            if !v.is_input() {
                continue;
            }
            if !moved && inside(v.get_addr(), v.get_size(), own, own_size) {
                ends.entry = true;
            } else {
                ends.values.push(cur);
            }
            continue;
        };
        let Some(op) = data.obank().get(def) else { continue };
        match op.code() {
            OpCode::CPUI_COPY if !op.get_in(0).and_then(|x| data.vbank().get(x)).is_some_and(|x| x.is_constant()) => {
                work.extend(op.get_in(0).map(|x| (x, moved || !moves_register_window(data, def))));
            }
            OpCode::CPUI_INDIRECT if op.is_indirect_creation() => {}
            OpCode::CPUI_INDIRECT => work.extend(op.get_in(0).map(|x| (x, moved))),
            OpCode::CPUI_MULTIEQUAL | OpCode::CPUI_PIECE => {
                work.extend((0..op.num_input()).filter_map(|i| op.get_in(i)).map(|x| (x, moved)));
            }
            OpCode::CPUI_SUBPIECE
                if op.get_in(0).and_then(|x| data.vbank().get(x)).is_some_and(|x| x.get_def().is_none()) =>
            {
                work.extend(op.get_in(0).map(|x| (x, moved)));
            }
            _ => ends.values.push(cur),
        }
    }
    ends
}

/// Does `[addr, addr + size)` lie inside the `own_size`-byte register at `own`?
fn inside(addr: &Address, size: int4, own: &Address, own_size: int4) -> bool {
    match (addr.get_space(), own.get_space()) {
        (Some(a), Some(b)) if Rc::ptr_eq(a, b) => {
            addr.get_offset() >= own.get_offset()
                && addr.get_offset() + size as u64 <= own.get_offset() + own_size as u64
        }
        _ => false,
    }
}

/// Is every use of `value` -- and of everything computed from it -- one of the
/// pair's RETURN slots: the low word itself, or the first register through the
/// value's sign (`sra 31`, an extension), a carry or borrow out of it, or the
/// other half of a wider value? Reaching the first register any other way, or
/// unchanged, makes it scratch; so does a call, a store, a branch or an address.
fn only_returned(data: &Funcdata, value: VarnodeId, pair: &Pair) -> bool {
    #[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
    enum Via {
        Same,
        Changed,
        Excused,
    }
    let bump = |via: Via| if via == Via::Excused { Via::Excused } else { Via::Changed };
    let mut seen: BTreeSet<(VarnodeId, Via)> = BTreeSet::new();
    let mut work = vec![(value, Via::Same)];
    while let Some((cur, via)) = work.pop() {
        if !seen.insert((cur, via)) {
            continue;
        }
        if seen.len() > MAX_NODES {
            return false;
        }
        let Some(v) = data.vbank().get(cur) else { continue };
        let bits = 8 * v.get_size() as u64;
        for d in v.descend_iter() {
            let Some(op) = data.obank().get(d) else { continue };
            if op.is_dead() {
                continue;
            }
            let next = match op.code() {
                OpCode::CPUI_RETURN => {
                    for i in 0..op.num_input() {
                        if op.get_in(i) != Some(cur) || i == pair.lo_slot {
                            continue;
                        }
                        if i != pair.hi_slot || via != Via::Excused {
                            return false;
                        }
                    }
                    continue;
                }
                OpCode::CPUI_CALL | OpCode::CPUI_CALLIND if cur == value && v.get_def().is_none() && undecided_input(data, d) => {
                    continue
                }
                OpCode::CPUI_CALL
                | OpCode::CPUI_CALLIND
                | OpCode::CPUI_CALLOTHER
                | OpCode::CPUI_STORE
                | OpCode::CPUI_CBRANCH
                | OpCode::CPUI_BRANCHIND => return false,
                OpCode::CPUI_LOAD => {
                    if op.get_in(1) == Some(cur) {
                        return false;
                    }
                    bump(via)
                }
                OpCode::CPUI_COPY | OpCode::CPUI_MULTIEQUAL => via,
                OpCode::CPUI_INDIRECT => {
                    if op.get_in(0) != Some(cur) {
                        continue;
                    }
                    via
                }
                OpCode::CPUI_SUBPIECE if v.get_size() > pair.lo_size => Via::Excused,
                OpCode::CPUI_INT_SEXT
                | OpCode::CPUI_INT_ZEXT
                | OpCode::CPUI_INT_EQUAL
                | OpCode::CPUI_INT_NOTEQUAL
                | OpCode::CPUI_INT_LESS
                | OpCode::CPUI_INT_LESSEQUAL
                | OpCode::CPUI_INT_SLESS
                | OpCode::CPUI_INT_SLESSEQUAL
                | OpCode::CPUI_INT_CARRY
                | OpCode::CPUI_INT_SCARRY
                | OpCode::CPUI_INT_SBORROW => Via::Excused,
                OpCode::CPUI_INT_SRIGHT
                    if op.get_in(0) == Some(cur)
                        && op
                            .get_in(1)
                            .and_then(|k| data.vbank().get(k))
                            .is_some_and(|k| k.is_constant() && k.get_offset() + 1 == bits) =>
                {
                    Via::Excused
                }
                _ => bump(via),
            };
            if let Some(out) = op.get_out() {
                work.push((out, next));
            }
        }
    }
    true
}

/// Are the inputs of `call` still trials -- registers a callee without a known
/// prototype might read -- rather than its settled arguments? The function's
/// own input passing through such a call is not yet an argument of it.
fn undecided_input(data: &Funcdata, call: OpId) -> bool {
    data.get_call_specs_index(call).is_some_and(|i| data.get_call_specs(i).is_input_active())
}

/// The value of `vn` when it is a constant or integer arithmetic and logic on
/// constants alone, within `depth` operations, truncated to its size.
fn literal_value(data: &Funcdata, vn: VarnodeId, depth: u32) -> Option<u64> {
    let v = data.vbank().get(vn)?;
    let size = v.get_size();
    let mask = |x: u64, sz: int4| if sz >= 8 { x } else { x & ((1u64 << (8 * sz)) - 1) };
    if v.is_constant() {
        return Some(mask(v.get_offset(), size));
    }
    if depth == 0 {
        return None;
    }
    let op = data.obank().get(v.get_def()?)?;
    let arg = |i: int4| op.get_in(i).and_then(|x| literal_value(data, x, depth - 1));
    let arg_size = |i: int4| op.get_in(i).and_then(|x| data.vbank().get(x)).map_or(0, |x| x.get_size());
    let r = match op.code() {
        OpCode::CPUI_COPY | OpCode::CPUI_INT_ZEXT => arg(0)?,
        OpCode::CPUI_INT_SEXT => {
            let (x, sz) = (arg(0)?, arg_size(0));
            if sz < 8 && x >> (8 * sz - 1) & 1 == 1 {
                x | !((1u64 << (8 * sz)) - 1)
            } else {
                x
            }
        }
        OpCode::CPUI_INT_ADD => arg(0)?.wrapping_add(arg(1)?),
        OpCode::CPUI_INT_SUB => arg(0)?.wrapping_sub(arg(1)?),
        OpCode::CPUI_INT_OR => arg(0)? | arg(1)?,
        OpCode::CPUI_INT_XOR => arg(0)? ^ arg(1)?,
        OpCode::CPUI_INT_AND => arg(0)? & arg(1)?,
        OpCode::CPUI_INT_LEFT => arg(0)?.checked_shl(arg(1)?.try_into().ok()?).unwrap_or(0),
        OpCode::CPUI_INT_RIGHT => arg(0)?.checked_shr(arg(1)?.try_into().ok()?).unwrap_or(0),
        OpCode::CPUI_INT_NEGATE => !arg(0)?,
        OpCode::CPUI_INT_2COMP => arg(0)?.wrapping_neg(),
        OpCode::CPUI_PIECE => {
            let lo_bits = 8 * arg_size(1) as u32;
            arg(0)?.checked_shl(lo_bits).unwrap_or(0) | arg(1)?
        }
        OpCode::CPUI_SUBPIECE => arg(0)?.checked_shr(8 * arg(1)? as u32).unwrap_or(0),
        _ => return None,
    };
    Some(mask(r, size))
}

/// Is `copy` one register's move in a register-window instruction: a COPY of
/// one register into another, at a machine instruction that copies every
/// general-purpose register the prototype model passes arguments in, either
/// out of them (SPARC's `save` moves `%o0`-`%o5` into `%i0`-`%i5`) or back
/// into them (`restore`)? The register copied may be a heritage temporary
/// reassembling a register the function also reads in parts. `restore`'s own
/// destination write (`restore %g0,1,%o1`) copies a temporary, so it is not the
/// window's. A compiler's move copies one register (`mov r1,r4`) or a pair
/// (AVR's `movw`), so it never answers `true`; neither does a model with fewer
/// than three argument registers, where "every" says too little.
fn moves_register_window(data: &Funcdata, copy: OpId) -> bool {
    fn is_register(a: &Address) -> bool {
        a.get_space().is_some_and(|sp| sp.get_type() == spacetype::IPTR_PROCESSOR)
    }
    fn reassembled(data: &Funcdata, src: &crate::varnode::Varnode, depth: u32) -> bool {
        depth > 0
            && src.get_def().and_then(|d| data.obank().get(d)).is_some_and(|piece| {
                piece.code() == OpCode::CPUI_PIECE
                    && (0..piece.num_input()).all(|i| {
                        piece
                            .get_in(i)
                            .and_then(|x| data.vbank().get(x))
                            .is_some_and(|x| is_register(x.get_addr()) || reassembled(data, x, depth - 1))
                    })
            })
    }
    let Some(op) = data.obank().get(copy) else { return false };
    let one_register = op.code() == OpCode::CPUI_COPY
        && match (op.get_out().and_then(|x| data.vbank().get(x)), op.get_in(0).and_then(|x| data.vbank().get(x))) {
            (Some(out), Some(src)) => {
                is_register(out.get_addr())
                    && out.get_addr() != src.get_addr()
                    && (is_register(src.get_addr()) || reassembled(data, src, 4))
            }
            _ => false,
        };
    if !one_register {
        return false;
    }
    let proto = data.get_func_proto();
    if !proto.has_model() {
        return false;
    }
    let Some(input) = proto.model().input_opt() else { return false };
    let at = op.get_addr().clone();
    let mut into: Vec<(Address, i32)> = Vec::new();
    let mut from: Vec<(Address, i32)> = Vec::new();
    for (_, id) in data.obank().iter_at(&at) {
        let Some(op) = data.obank().get(id) else { continue };
        if op.is_dead() || op.code() != OpCode::CPUI_COPY {
            continue;
        }
        let (Some(out), Some(src)) =
            (op.get_out().and_then(|x| data.vbank().get(x)), op.get_in(0).and_then(|x| data.vbank().get(x)))
        else {
            continue;
        };
        if src.get_addr() == out.get_addr() {
            continue;
        }
        if is_register(out.get_addr()) {
            into.push((out.get_addr().clone(), out.get_size()));
        }
        if is_register(src.get_addr()) {
            from.push((src.get_addr().clone(), src.get_size()));
        }
    }
    let registers: Vec<&crate::fspec::ParamEntry> = input
        .get_entry()
        .iter()
        .filter(|e| {
            e.get_type() == crate::dtype::type_class::TYPECLASS_GENERAL
                && e.get_space().get_type() == spacetype::IPTR_PROCESSOR
        })
        .collect();
    let covers = |moved: &[(Address, i32)]| {
        registers.iter().all(|e| {
            moved.iter().any(|(a, size)| {
                *size == e.get_size()
                    && a.get_offset() == e.get_base()
                    && a.get_space().is_some_and(|sp| Rc::ptr_eq(sp, e.get_space()))
            })
        })
    };
    registers.len() >= 3 && (covers(&into) || covers(&from))
}

/// Is `retop` reached only along branch edges whose conditions literals
/// already decide the other way? SPARC's `call` keeps such a RETURN for a
/// `restore` in its delay slot (`didrestore = 0; ...; if (didrestore == 0)
/// goto next; return [o7]`), and return recovery settles the prototype before
/// the rule pool folds it away, so its `%o1` is whatever the call was passed.
fn never_reached(data: &Funcdata, retop: OpId) -> bool {
    let Some(bl) = data.obank().get(retop).and_then(|o| o.get_parent()) else { return false };
    let block = data.bblocks_ref().block(bl);
    if block.size_in() == 0 {
        return false;
    }
    (0..block.size_in()).all(|k| {
        let (from, edge) = (block.get_in(k), block.get_in_rev_index(k));
        let Some(cbranch) = data.bb_op_tail(from).and_then(|t| data.obank().get(t)) else { return false };
        if cbranch.code() != OpCode::CPUI_CBRANCH {
            return false;
        }
        let Some(val) = cbranch.get_in(1).and_then(|c| decided(data, c, 8)) else { return false };
        let taken = if (val != 0) != cbranch.is_boolean_flip() { 1 } else { 0 };
        taken != edge
    })
}

/// The value of a condition built from literals by copies, `==`, `!=` and
/// `!`, or `None` when anything else (an INDIRECT a call may change, an input)
/// feeds it.
fn decided(data: &Funcdata, vn: VarnodeId, depth: u32) -> Option<u64> {
    let v = data.vbank().get(vn)?;
    if v.is_constant() {
        return Some(v.get_offset());
    }
    if depth == 0 {
        return None;
    }
    let op = data.obank().get(v.get_def()?)?;
    let arg = |i: i32| op.get_in(i).and_then(|x| decided(data, x, depth - 1));
    match op.code() {
        OpCode::CPUI_COPY => arg(0),
        OpCode::CPUI_BOOL_NEGATE => arg(0).map(|x| (x == 0) as u64),
        OpCode::CPUI_INT_EQUAL => Some((arg(0)? == arg(1)?) as u64),
        OpCode::CPUI_INT_NOTEQUAL => Some((arg(0)? != arg(1)?) as u64),
        _ => None,
    }
}

/// Does `whole` sit in two of the model's output registers, the FIRST of them
/// holding its most significant half -- a pair return recovery joined in the
/// ABI's order ([`join_order`])? `false` for one register, storage the model
/// does not list, or a pair joined first register low.
pub(crate) fn first_register_holds_high(data: &Funcdata, whole: VarnodeId) -> bool {
    pair_halves(data, whole)
        .is_some_and(|(hi, hi_size, lo, lo_size)| data.get_func_proto().output_holds_high_first(&hi, hi_size, &lo, lo_size))
}

/// The storage of `whole`'s most and least significant halves -- two pieces of
/// a join, or the two halves of one register -- with their sizes.
fn pair_halves(data: &Funcdata, whole: VarnodeId) -> Option<(Address, i32, Address, i32)> {
    use crate::kuna_returnuncomputed::{slot_storage, storage_pieces};
    let pieces = storage_pieces(data, whole)?;
    match pieces.as_slice() {
        [(hs, ho, hz), (ls, lo, lz)] => {
            Some((Address::new(Rc::clone(hs), *ho), *hz, Address::new(Rc::clone(ls), *lo), *lz))
        }
        [_] => {
            let size = data.vbank().get(whole).map(|v| v.get_size()).unwrap_or(0);
            if size < 2 || size % 2 != 0 {
                return None;
            }
            let half = size / 2;
            Some((slot_storage(data, whole, half, half)?, half, slot_storage(data, whole, 0, half)?, half))
        }
        _ => None,
    }
}

#[cfg(test)]
#[path = "kuna_bejoin/tests.rs"]
mod tests;
