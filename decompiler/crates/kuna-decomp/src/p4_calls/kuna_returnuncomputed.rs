//! (kuna, ida) Reject a return register that carries no value the function ever
//! computed — the `undefined16 main(...)` / `return v4;` phantom.
//!
//! # The symptom
//!
//! An x86-64 SysV function with no known prototype was modelled as returning a
//! **16-byte** value, materialized by writing the genuine result to byte 0 and an
//! unrelated leftover to byte 8:
//!
//! ```text
//! char v4 [16];
//! v4[0] = v16 ^ 1;   // the real int result
//! v4[8] = v22;       // an uninitialized stack slot
//! return v4;
//! ```
//!
//! That output is not merely unreadable, it is wrong: it reads memory the
//! function never wrote. IDA Pro recovers `return (unsigned __int8)v16 ^ 1;`.
//!
//! # Why it happens
//!
//! Return recovery registers one trial per output register the prototype model
//! characterizes — for x86-64 gcc that is `RAX` *and* `RDX` — and marks a trial
//! active when its value survives ancestor-realism and is used only at the
//! RETURN. The compiler spec's output rule (`join_dual_class`) then accepts two
//! consecutive active trials as one 16-byte return, and the RETURN is rewritten
//! to `PIECE(RDX,RAX)`.
//!
//! Ancestor realism asks whether a value could *legitimately reach* the RETURN —
//! it is not asking whether the function meant to return it. Both of the shapes
//! that produce the phantom pass it:
//!
//! * a **callee-saved register restore**: the epilogue's `RDX` is a copy of an
//!   input Varnode for a stack slot the function never stores to, so the value is
//!   whatever the caller's frame happened to hold;
//! * a **clobber at a no-return call**: where the flow model turns a call that
//!   never returns into a return, `RDX` is the INDIRECT-creation standing for
//!   "the callee wrote something here", which is a statement about the callee,
//!   not a value.
//!
//! The upstream port already rejects the second shape *when it sees it* (a trial
//! formed from an INDIRECT creation is dropped unless it is first in its storage
//! class), but a trial is only checked once, at the first live RETURN — so a
//! function that reaches the restore shape first keeps the trial and joins.
//!
//! # When the decision is made
//!
//! Not at recovery time. There, the epilogue's restore is still
//! `COPY(LOAD(sp - k))` — indistinguishable from `return *p` — so there is
//! nothing to decide on. By the time the prototype is fixated, heritage has
//! resolved that load into a bare unwritten Varnode for a frame slot the function
//! never stores to, and the difference is plain. So this runs late, on the
//! already-built concatenation, and rewrites the RETURN to the half that carries
//! a value.
//!
//! # The rule
//!
//! A return trial whose value, at **every** live RETURN, traces back only to
//! things the function did not compute is not a return value.
//!
//! An unwritten Varnode that is a **formal input parameter** is the one exception
//! — the function was handed that value, so handing it back is a real return.
//! That carve-out is `option retinputhalf` and lives in
//! [`crate::kuna_retinputhalf`]; without it a returned pair whose high half is a
//! passthrough argument loses the half *and* the argument.
//!
//! "Did not compute" is decided by a bounded walk back through the operations
//! that only *move* a value — copies, phis, indirects, and piece/subpiece
//! reshaping — stopping at the first operation that produces one. A terminal is
//! uncomputed when it is an unwritten (input or free) Varnode or an INDIRECT
//! creation; a constant counts as computed, because returning a literal is a real
//! return. Anything the walk cannot classify is treated as computed, so an
//! unfamiliar shape keeps today's answer.
//!
//! # A value built in one return register
//!
//! The same `PIECE` appears when the function assembles a value in its single
//! return register: `((u64)hi << 32) | lo` folds into `RAX = PIECE(ESI, EDI)`,
//! and heritage splits a partly written register into `RAX = PIECE(RAX[4:4],
//! EAX)`. Each half is therefore judged against the bytes it occupies in the
//! returned storage (its register of a join, or its offset in the one register),
//! never against its own address -- an argument folded straight into the value
//! sits at its own address by definition and would read as leftover, dropping
//! the half, narrowing the return and orphaning the argument. And the high half
//! of one register is never kept alone: that would return its bits as the whole
//! value.
//!
//! # Why this cannot break a genuine multi-register return
//!
//! Two independent guards. First, the rule only ever edits a value concatenated
//! from two halves, and never deactivates the last survivor — so a function with
//! a single recovered return register is untouched, whatever its value looks like.
//! Second, a real 16-byte struct return *computes* both halves: it builds them
//! from constants, arithmetic, or loads through a pointer, and a LOAD is not in
//! the move-only set, so the walk stops there and reports computed. Only a half
//! that is pure leftover — never written, or a callee's clobber — is dropped.

use std::rc::Rc;

use kuna_base::address::Address;
use kuna_base::space::{spacetype, AddrSpace};
use kuna_num::opcodes::OpCode;

use crate::context::{OpId, VarnodeId};
use crate::fspec::ParamActive;
use crate::funcdata::Funcdata;

/// How far back the walk chases move-only operations before giving up and
/// calling the value computed. Deep enough for the epilogue chains that produce
/// the phantom (a restore is a copy or two; a clobber is one indirect), shallow
/// enough that this never shows up in a profile.
const MAX_DEPTH: u32 = 24;

/// How many Varnodes the whole-function walk ([`computes_everywhere`]) visits
/// before giving up and answering `false` -- "not computed", which is the
/// refusal: the callee's recovered return is not stated, and every caller reads
/// the call exactly as it does with `passthrough` off. Exhausting it takes a
/// move-only closure of 4,096 Varnodes reachable from ONE returned value, which
/// is two orders of magnitude past the deepest copy/phi chain measured over the
/// decbench corpus, so the cap is a guard against a pathological body rather
/// than a budget the walk is expected to spend.
const MAX_NODES: u32 = 4096;

/// [`computes_from`], asked of every byte and every path instead of any one of
/// them: a value is computed only when NO terminal reachable through the
/// move-only operations is one the function never produced.
///
/// The relaxed question is the right one for the pair repair, which asks it of
/// one half at a time and has to keep a genuine wide return. A caller reading a
/// callee's recovered return type needs the strict one: a value pieced together
/// from a call's narrow result and a leftover, or merged from one path that
/// computes it and one that does not, is not a return value the caller can hand
/// on.
///
/// Written as a worklist over the reachable move-only closure rather than the
/// recursion [`computes_from`] uses: "every input" over a phi-rich -O0 body
/// revisits the same Varnodes exponentially, and this runs once per decompiled
/// function instead of once per returned register pair.
///
/// Running out of budget answers `false`, not `true`: the answer is read by
/// [`crate::p4_calls::kuna_protoorder`]'s `recovered_output`, where `true` states
/// the callee's return to every caller ahead of it and `false` states nothing at
/// all, so the unfinished walk has to take the second.
fn computes_everywhere(data: &Funcdata, vn: VarnodeId, placed_at: Option<&Address>, globals: bool) -> bool {
    let mut seen: std::collections::HashSet<VarnodeId> = std::collections::HashSet::new();
    let mut work: Vec<VarnodeId> = vec![vn];
    let mut budget = MAX_NODES;
    while let Some(cur) = work.pop() {
        if budget == 0 {
            // `true` here would STATE the callee's return on a body the walk
            // never finished reading, which is the direction that invents one.
            return false;
        }
        budget -= 1;
        if !seen.insert(cur) {
            continue;
        }
        let Some(v) = data.vbank().get(cur) else { continue };
        if v.is_constant() {
            continue;
        }
        let Some(def) = v.get_def() else {
            if placed_at.is_some_and(|a| a == v.get_addr()) {
                return false;
            }
            if globals && v.is_persist() {
                continue;
            }
            if !crate::kuna_retinputhalf::is_input_parameter(data, cur) {
                return false;
            }
            continue;
        };
        let Some(op) = data.obank().get(def) else { continue };
        if op.code() == OpCode::CPUI_INDIRECT && op.is_indirect_creation() {
            return false;
        }
        match op.code() {
            OpCode::CPUI_COPY | OpCode::CPUI_INDIRECT | OpCode::CPUI_SUBPIECE => {
                work.extend(op.get_in(0));
            }
            OpCode::CPUI_MULTIEQUAL | OpCode::CPUI_PIECE => {
                work.extend((0..op.num_input()).filter_map(|i| op.get_in(i)));
            }
            // Anything else produces a value; the walk stops here.
            _ => continue,
        }
    }
    true
}

/// The walk, carrying the input-parameter carve-out's **placement** test.
///
/// `placed_at` is the storage the half occupies in the returned value, which
/// turns the carve-out into "did the function PUT an argument here": a terminal at
/// a different address was moved into the return register by an instruction the
/// function executed, while a terminal at the same address is the caller's
/// register passing straight through untouched -- leftover, and exactly what this
/// module exists to drop. `None` drops the placement test and is the shape-only
/// question the unit tests ask. See [`crate::kuna_retinputhalf`].
fn computes_from(data: &Funcdata, vn: VarnodeId, depth: u32, placed_at: Option<&Address>) -> bool {
    if depth >= MAX_DEPTH {
        return true;
    }
    let Some(v) = data.vbank().get(vn) else { return true };
    // A literal IS a computed return: `return 0;` is a return value.
    if v.is_constant() {
        return true;
    }
    // Never written: a function input, or a free Varnode standing for a location
    // the function only ever reads (the callee-saved restore shape). An input
    // parameter the function PLACED in the return register is a value it was
    // handed and is handing back, which is a real return.
    let Some(def) = v.get_def() else {
        if placed_at.is_some_and(|a| a == v.get_addr()) {
            return false;
        }
        return crate::kuna_retinputhalf::is_input_parameter(data, vn);
    };
    let Some(op) = data.obank().get(def) else { return true };
    // The callee wrote something here; that is a fact about the callee.
    if op.code() == OpCode::CPUI_INDIRECT && op.is_indirect_creation() {
        return false;
    }
    let inputs: Vec<VarnodeId> = match op.code() {
        // Pure moves: chase the source.
        OpCode::CPUI_COPY | OpCode::CPUI_INDIRECT | OpCode::CPUI_SUBPIECE => {
            op.get_in(0).into_iter().collect()
        }
        // Reshaping and phis: computed if ANY input is.
        OpCode::CPUI_MULTIEQUAL | OpCode::CPUI_PIECE => {
            (0..op.num_input()).filter_map(|i| op.get_in(i)).collect()
        }
        // Anything else produces a value.
        _ => return true,
    };
    if inputs.is_empty() {
        return true;
    }
    inputs.into_iter().any(|i| computes_from(data, i, depth + 1, placed_at))
}

/// Does every live RETURN of `data` hand back a value the function computed?
///
/// The same walk the pair repair runs, asked of the whole function and of a
/// single return register, where the repair cannot act: it only ever chooses
/// between two active trials and never deactivates the last survivor, so a
/// function recovered with ONE return register keeps whatever reached the
/// RETURN. gnulib's `version_etc_arn` is `void`, ends its fallthrough path in a
/// `__fprintf_chk` and keeps that call's `RAX` clobber -- an INDIRECT creation --
/// as its "result", so kuna recovers it as returning `long`. A caller has no way
/// to see that from the recovered prototype alone, which is why `passthrough`
/// asks this question of the callee (see [`crate::kuna_passthrough`]).
///
/// `false` means at least one RETURN hands back a terminal the function never
/// computed. A function with no live RETURN answers `true`, the no-change answer.
pub fn every_return_computes(data: &Funcdata) -> bool {
    every_return_computes_with(data, false)
}

/// [`every_return_computes`], with a value read out of global memory
/// (`return stdout;`) counted as computed when `globals`: the program's own
/// data, not a register or frame slot the caller left behind.  `callrettype`
/// asks it this way ([`crate::p4_calls::kuna_callrettype`]).
pub fn every_return_computes_with(data: &Funcdata, globals: bool) -> bool {
    for retop in data.obank().iter_code(OpCode::CPUI_RETURN).collect::<Vec<_>>() {
        let Some(o) = data.obank().get(retop) else { continue };
        if o.is_dead() || o.get_halt_type() != 0 || o.num_input() < 2 {
            continue;
        }
        let Some(value) = o.get_in(1) else { continue };
        let placed = data.vbank().get(value).map(|v| v.get_addr().clone());
        if !computes_everywhere(data, value, placed.as_ref(), globals) {
            return false;
        }
    }
    true
}

/// The storage locations `vn` occupies, most significant first: the pieces of
/// a join, or `vn`'s own storage.
fn storage_pieces(data: &Funcdata, vn: VarnodeId) -> Option<Vec<(Rc<AddrSpace>, u64, i32)>> {
    let v = data.vbank().get(vn)?;
    let addr = v.get_addr();
    let space = addr.get_space()?;
    if space.get_type() != spacetype::IPTR_JOIN {
        return Some(vec![(Rc::clone(space), addr.get_offset(), v.get_size())]);
    }
    let rec = data.get_arch().manage().find_join(addr.get_offset()).ok()?;
    (0..rec.num_pieces())
        .map(|i| {
            let p = rec.get_piece(i);
            p.space.clone().map(|s| (s, p.offset, p.size as i32))
        })
        .collect()
}

/// Is `vn` stored across two or more locations (a register pair's join)?
fn spans_two_locations(data: &Funcdata, vn: VarnodeId) -> bool {
    storage_pieces(data, vn).is_some_and(|p| p.len() > 1)
}

/// Where the `width` bytes of `whole` starting `lsb` bytes above its least
/// significant byte are stored: the return register (or register of a pair) a
/// half of the returned value sits in. `None` when they straddle two pieces.
fn slot_storage(data: &Funcdata, whole: VarnodeId, lsb: i32, width: i32) -> Option<Address> {
    let mut lsb = lsb;
    for (space, off, size) in storage_pieces(data, whole)?.into_iter().rev() {
        if lsb < size {
            if lsb + width > size {
                return None;
            }
            let rel = if space.is_big_endian() { size - lsb - width } else { lsb };
            let at = space.wrap_offset(off.wrapping_add(rel as u64));
            return Some(Address::new(space, at));
        }
        lsb -= size;
    }
    None
}

/// Repair a RETURN whose value is a return-recovery register **pair** with an
/// uncomputed half: rewrite it to the half that carries a value, and destroy the
/// now-dead concatenation.
///
/// Runs late, in the one-shot tail, and that timing is the whole point. When
/// return recovery makes the pair decision the epilogue's restore still looks
/// like `COPY(LOAD(sp - k))` — indistinguishable from `return *p` — so there is
/// nothing to decide on. By the time the prototype is fixated, heritage has
/// resolved that load into a bare unwritten Varnode for a frame slot the function
/// never stores to, and the difference is plain.
///
/// A register-window pair ([`WindowPair`]) is repaired at every live RETURN
/// or at none: when one RETURN keeps both registers, the others keep them too,
/// since the function returns one value everywhere.
///
/// Returns `true` when a RETURN was rewritten.
pub fn strip_uncomputed_return_piece(data: &mut Funcdata) -> bool {
    // Collect first: the rewrite mutates the op bank.
    let mut fixes: Vec<(OpId, Kept, OpId)> = Vec::new();
    let mut window_fixes: Vec<(OpId, Kept, OpId)> = Vec::new();
    let mut window_kept = false;
    let window = data.kuna_window_pair();
    for retop in data.obank().iter_code(OpCode::CPUI_RETURN).collect::<Vec<_>>() {
        let Some(o) = data.obank().get(retop) else { continue };
        if o.is_dead() || o.get_halt_type() != 0 || o.num_input() < 2 {
            continue;
        }
        let Some(joined) = o.get_in(1) else { continue };
        let high_first = first_register_holds_high(data, joined);
        let in_window = high_first && window != WindowPair::No;
        match repair_return(data, joined, high_first, window) {
            Some((keep, def)) if in_window => window_fixes.push((retop, keep, def)),
            Some((keep, def)) => fixes.push((retop, keep, def)),
            None => window_kept |= in_window,
        }
    }
    if !window_kept {
        fixes.append(&mut window_fixes);
    }

    if fixes.is_empty() {
        return false;
    }
    let mut scratch: Vec<OpId> = Vec::new();
    let mut pieces: Vec<OpId> = Vec::new();
    for (retop, keep, piece) in fixes {
        let keep = match keep {
            Kept::Half(vn) => vn,
            Kept::Literal(size, value) => data.new_constant(size, value),
        };
        if data.op_set_input(retop, keep, 1).is_ok() {
            pieces.push(piece);
        }
    }
    // Every RETURN reads its kept half before any concatenation goes, so one
    // RETURN's cleanup never frees a Varnode another RETURN now reads.
    for piece in pieces {
        // The concatenation now has no readers. Destroy it so the printer does
        // not emit the phantom `v[8] = <leftover>` write that materialized it.
        let unused = data
            .obank()
            .get(piece)
            .filter(|o| !o.is_dead())
            .and_then(|o| o.get_out())
            .and_then(|v| data.vbank().get(v))
            .map(|v| v.has_no_descend())
            .unwrap_or(false);
        if unused {
            data.op_destroy_recursive(piece, &mut scratch);
        }
    }
    true
}

/// What [`strip_uncomputed_return_piece`] returns instead of a pair.
enum Kept {
    /// One half of the pair.
    Half(VarnodeId),
    /// The high half of a literal pair: its size and value.
    Literal(i32, u64),
}

/// What [`strip_uncomputed_return_piece`] hands one RETURN whose value is
/// `joined`, and the op that built the pair; `None` to keep the pair.
fn repair_return(data: &Funcdata, joined: VarnodeId, high_first: bool, window: WindowPair) -> Option<(Kept, OpId)> {
    let def = data.vbank().get(joined).and_then(|v| v.get_def())?;
    let top_code = data.obank().get(def)?.code();
    if high_first && window != WindowPair::No {
        if let Some(high) = shifted_high_half(data, joined) {
            return Some((Kept::Half(high), def));
        }
        if let Some((value, hi_size, lo_size)) = literal_pair(data, joined) {
            let low = if lo_size >= 8 { value } else { value & ((1u64 << (8 * lo_size)) - 1) };
            if low == 0 || window == WindowPair::Leftover {
                let high = if lo_size >= 8 { 0 } else { value >> (8 * lo_size) };
                return Some((Kept::Literal(hi_size, high), def));
            }
        }
    }
    // When the first register holds the high half, the value sits there,
    // where the rule pool folds its zero-extension out of the join:
    // ZEXT(PIECE(x, lo)). Look through it; the kept high half is `x` itself.
    let piece_op = if high_first && top_code == OpCode::CPUI_INT_ZEXT {
        data.obank().get(def)?.get_in(0).and_then(|v| data.vbank().get(v)).and_then(|v| v.get_def())?
    } else {
        def
    };
    let piece = data.obank().get(piece_op)?;
    // Only the two-register join return recovery builds; anything else is
    // someone else's op and stays.
    if piece.code() != OpCode::CPUI_PIECE || piece.num_input() != 2 {
        return None;
    }
    let (hi, lo) = (piece.get_in(0)?, piece.get_in(1)?);
    let whole = joined;
    let (h, l) = (data.vbank().get(hi)?, data.vbank().get(lo)?);
    let (hi_addr, hi_size, lo_addr, lo_size) = (h.get_addr().clone(), h.get_size(), l.get_addr().clone(), l.get_size());
    let hi_slot = slot_storage(data, whole, lo_size, hi_size).unwrap_or(hi_addr);
    let lo_slot = slot_storage(data, whole, 0, lo_size).unwrap_or(lo_addr);
    let hi_real = computes_from(data, hi, 0, Some(&hi_slot));
    let lo_real = !(high_first && window == WindowPair::Leftover) && computes_from(data, lo, 0, Some(&lo_slot));
    let keep = match (hi_real, lo_real) {
        // Both halves carry a value: a genuine wide return. Leave it alone.
        (true, true) => return None,
        // One return register holds both halves, so its high bits are not a
        // return value of their own: handing them back alone would return
        // them in place of the whole register.
        (true, false) if !high_first && !spans_two_locations(data, whole) => return None,
        (true, false) => hi,
        // Only the low half is real — the common case, a callee-saved restore
        // in the high register.
        (false, true) => lo,
        // Neither half is real: this is the return the flow model synthesizes
        // where a call that never returns falls through, and both registers
        // hold the callee's clobber. There is no return value to recover, but
        // the function's output storage has to agree across every RETURN, so
        // keep the first-in-class register — what the model would have
        // picked had the join never formed. A pair joined first register
        // high holds it as the high half; one register's high bits are
        // never a return value.
        (false, false) if high_first => hi,
        (false, false) => lo,
    };
    Some((Kept::Half(keep), def))
}

/// The first register's value in `whole`, a register-window pair whose low
/// half the rule pool proved zero and folded into `ZEXT(x) << lo`: `x`, when it
/// is exactly the high half.
fn shifted_high_half(data: &Funcdata, whole: VarnodeId) -> Option<VarnodeId> {
    let (_, hi_size, _, lo_size) = pair_halves(data, whole)?;
    let shift = data.obank().get(data.vbank().get(whole)?.get_def()?)?;
    if shift.code() != OpCode::CPUI_INT_LEFT {
        return None;
    }
    let by = data.vbank().get(shift.get_in(1)?)?;
    if !by.is_constant() || by.get_offset() != 8 * lo_size as u64 {
        return None;
    }
    let ext = data.obank().get(data.vbank().get(shift.get_in(0)?)?.get_def()?)?;
    if ext.code() != OpCode::CPUI_INT_ZEXT {
        return None;
    }
    let x = ext.get_in(0)?;
    (data.vbank().get(x)?.get_size() == hi_size).then_some(x)
}

/// `whole`, a pair of registers, when it holds a literal (a copy of one, the
/// shape the rule pool folds a `PIECE` of two literals into): the value and
/// the high and low halves' sizes.
fn literal_pair(data: &Funcdata, whole: VarnodeId) -> Option<(u64, i32, i32)> {
    let (_, hi_size, _, lo_size) = pair_halves(data, whole)?;
    let copy = data.obank().get(data.vbank().get(whole)?.get_def()?)?;
    if copy.code() != OpCode::CPUI_COPY {
        return None;
    }
    let k = data.vbank().get(copy.get_in(0)?)?;
    k.is_constant().then(|| (k.get_offset(), hi_size, lo_size))
}

/// Classify the second register of a pair the output rule joined first
/// register high, for [`narrow_window_pair`]: what a register window hands
/// back in it.
///
/// SPARC is where this matters. `save` copies every out-register into its
/// in-register and `restore` copies them back, so `%o1` reaches the RETURN
/// holding whatever the function last put in `%i1`: the incoming second
/// argument when it never touched it, a literal when it used `%i1` as scratch
/// (clang's `mov %g0,%i1` before a byte store, an address half `sethi` built),
/// or a value it computed. A leaf function that leaves `%o1` untouched has its
/// second trial rejected outright (an unmodified input is not a return value);
/// the window's copies make the same register look moved, so this asks the
/// leaf function's question again with those copies seen through.
///
/// Each live RETURN's second register is judged on its own ([`window_value`]);
/// a RETURN reached only along branches literals decide the other way is
/// skipped ([`never_reached`]).
///
/// * [`WindowPair::No`] when at some RETURN the function wrote the register
///   itself -- through `restore`'s destination (`restore %g0,1,%o1`), which
///   is not a window move -- or put a value in `%i1` for the window to hand
///   back and used it for nothing else ([`HandedValue::Deliberate`]).
/// * [`WindowPair::Leftover`] otherwise, when at some RETURN it is a leftover:
///   a function whose low word is a leftover on one path returns no low word
///   (an `int` function that hands back the untouched argument on its error
///   path and a loop counter on the other).
/// * [`WindowPair::HandedBack`] otherwise: at every RETURN it holds a value
///   the function computed and also used.
///
/// A register a compiler moves on purpose is not a window's. ARM big-endian's
/// `mov r4,r1; bl ext; mov r1,r4` carries the second argument across a call
/// into the low word of a returned `long long` -- the same data flow as the
/// window's, but through instructions that move one register, so it is a
/// return value. So is a literal put in a leaf function's `%o1` directly.
pub fn classify_window_pair(active: &ParamActive, data: &Funcdata, return_ops: &[OpId]) -> WindowPair {
    if !active.is_join_reverse() || active.get_num_trials() < 2 {
        return WindowPair::No;
    }
    let (first, second) = (active.get_trial(0), active.get_trial(1));
    if !first.is_used() || !second.is_used() {
        return WindowPair::No;
    }
    if active.get_num_trials() > 2 && active.get_trial(2).is_used() {
        return WindowPair::No;
    }
    let (slot, addr) = (second.get_slot(), second.get_address().clone());
    let mut returned: Vec<VarnodeId> = Vec::new();
    for &retop in return_ops {
        let Some(o) = data.obank().get(retop) else { continue };
        if o.is_dead() || o.get_halt_type() != 0 || never_reached(data, retop) {
            continue;
        }
        let Some(v) = o.get_in(slot) else { return WindowPair::No };
        let restored = data.vbank().get(v).and_then(|x| x.get_def()).is_some_and(|d| moves_register_window(data, d));
        if !restored {
            return WindowPair::No;
        }
        returned.push(v);
    }
    if returned.is_empty() {
        return WindowPair::No;
    }
    let values = window_values(data, &returned, &addr, first.get_slot(), &mut |copy| moves_register_window(data, copy));
    if values.contains(&HandedValue::Deliberate) {
        WindowPair::No
    } else if values.contains(&HandedValue::Leftover) {
        WindowPair::Leftover
    } else {
        WindowPair::HandedBack
    }
}

/// What [`classify_window_pair`] learned about a register-window
/// pair's second register, for [`narrow_window_pair`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum WindowPair {
    /// Not a window pair, or the function returns the register on purpose.
    #[default]
    No,
    /// A window hands the register back at every RETURN, holding a value the
    /// function computed and also used.
    HandedBack,
    /// A window hands the register back at every RETURN, holding its entry
    /// value, or a literal the function also used, at one of them at least.
    Leftover,
}

/// What a register window hands back in the second register at one RETURN
/// ([`window_values`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum HandedValue {
    /// The value the register held on entry, a literal the function also
    /// used for something else (clang's `mov %g0,%i1` before a byte store), or
    /// a value the function built its first register from (`mov 2,%i1; ret;
    /// restore %g0,%i1,%o0`).
    Leftover,
    /// A value the function computed and also used (a loop's pointer, the
    /// last byte a scan tested, a running total).
    Computed,
    /// A literal or a value the function put in the in-register for the window
    /// to hand back, and used for nothing else: `mov 10,%i1; ret; restore`.
    Deliberate,
}

/// Where a register-window pair's first register sits, read off the first
/// live RETURN, while the function still returns the pair: the output storage
/// when the late repair hands the RETURN a literal, whose own address is in the
/// constant space.
pub fn window_high_storage(data: &Funcdata) -> Option<Address> {
    if data.kuna_window_pair() == WindowPair::No {
        return None;
    }
    let retop = data.get_first_return_op()?;
    let whole = data.obank().get(retop)?.get_in(1)?;
    let (hi, hi_size, lo, lo_size) = pair_halves(data, whole)?;
    data.get_func_proto().output_holds_high_first(&hi, hi_size, &lo, lo_size).then_some(hi)
}

/// Replace the second register of a leftover window pair with zero in each
/// RETURN's join whose first register returns zero.
///
/// A leftover (the entry value or a literal, [`WindowPair::Leftover`]) is not
/// part of the value, and next to a zero first register it decides what the
/// rule pool folds the join into: `PIECE(0, a1)` becomes `ZEXT(a1)`, which the
/// late repair cannot tell from returning the argument (and which a later pass
/// narrows to the argument's own register). With a zero low half the join folds
/// into a literal, which [`strip_uncomputed_return_piece`] reads. A first
/// register can become zero on one path only once conditional constant
/// propagation has run, so return recovery calls this right after the join and
/// again at the start of every later pass, before the rule pool sees the join.
/// Other joins keep their leftover: the late repair keeps their first register
/// from the `PIECE` itself, typed as its value, and the RETURN keeps reading the
/// whole pair until then, so the first register's value is not made one
/// variable with the other values SPARC's unrelocated calls leave in `%o0`.
///
/// Returns `true` when a join was changed.
pub fn zero_leftover_low_half(data: &mut Funcdata) -> bool {
    if data.kuna_window_pair() != WindowPair::Leftover {
        return false;
    }
    let mut pieces: Vec<(OpId, i32)> = Vec::new();
    for retop in data.obank().iter_code(OpCode::CPUI_RETURN) {
        let Some(o) = data.obank().get(retop) else { continue };
        if o.is_dead() || o.get_halt_type() != 0 || o.num_input() != 2 {
            continue;
        }
        let Some(whole) = o.get_in(1) else { continue };
        let Some(piece) = data.vbank().get(whole).and_then(|v| v.get_def()) else { continue };
        let Some(op) = data.obank().get(piece) else { continue };
        if op.code() != OpCode::CPUI_PIECE || op.num_input() != 2 || !first_register_holds_high(data, whole) {
            continue;
        }
        let Some(lo) = op.get_in(1).and_then(|l| data.vbank().get(l)) else { continue };
        let Some(hi) = op.get_in(0) else { continue };
        let hi_bits = data.vbank().get(hi).map_or(0, |h| 8 * h.get_size() as u32);
        if !lo.is_constant() && low_bits_zero(data, hi, hi_bits, ZERO_DEPTH) {
            pieces.push((piece, lo.get_size()));
        }
    }
    let changed = !pieces.is_empty();
    for (piece, size) in pieces {
        let zero = data.new_constant(size, 0);
        let _ = data.op_set_input(piece, zero, 1);
    }
    changed
}

/// Narrow a register-window pair to its first register when the rule pool
/// has folded it into a shape the late repair cannot read.
///
/// The late repair [`strip_uncomputed_return_piece`] can only point the RETURN
/// at a Varnode that exists, or a literal: it reads a window pair's `PIECE`,
/// `ZEXT(x) << 32` (the shape a low half proved zero folds into) and a
/// literal. Return recovery calls this on every pass of the main loop once the
/// pair is built; when some RETURN has another shape (`CONCAT44(x, a1) &
/// 0xffffffffffff`, `(ZEXT(x) << 32) ^ k`) and the second register is a
/// leftover, or the low half of every returned value is zero whatever the
/// inputs hold ([`low_bits_zero`]), each RETURN is given `SUBPIECE(whole, lo)`
/// in the first register instead. A register the function worked with is often zero
/// at its exit by the control flow alone -- the pointer a
/// `for (p = head; p; p = next)` loop leaves on, the last byte a scan tested --
/// which only conditional constant propagation shows, so this is asked on every
/// pass. As for a literal, a genuine `(u64)x << 32` returned through a window
/// prints as `x`.
///
/// Returns `true` when the RETURNs were rewritten.
pub fn narrow_window_pair(data: &mut Funcdata) -> bool {
    let pair = data.kuna_window_pair();
    if pair == WindowPair::No {
        return false;
    }
    let mut plan: Vec<(OpId, VarnodeId, Address, i32, i32)> = Vec::new();
    let (mut low_zero, mut odd_shape) = (true, false);
    for retop in data.obank().iter_code(OpCode::CPUI_RETURN).collect::<Vec<_>>() {
        let Some(o) = data.obank().get(retop) else { continue };
        if o.is_dead() || o.get_halt_type() != 0 {
            continue;
        }
        if o.num_input() != 2 {
            return false;
        }
        let Some(whole) = o.get_in(1) else { return false };
        let Some((hi, hi_size, lo, lo_size)) = pair_halves(data, whole) else { return false };
        if !data.get_func_proto().output_holds_high_first(&hi, hi_size, &lo, lo_size) {
            return false;
        }
        if !never_reached(data, retop) {
            low_zero = low_zero && low_bits_zero(data, whole, 8 * lo_size as u32, ZERO_DEPTH);
            odd_shape |= !late_strippable(data, whole, lo_size);
        }
        plan.push((retop, whole, hi, hi_size, lo_size));
    }
    if plan.is_empty() || !odd_shape || !(low_zero || pair == WindowPair::Leftover) {
        return false;
    }
    data.kuna_set_window_pair(WindowPair::No);
    for (retop, whole, hi, hi_size, lo_size) in plan {
        let Some(at) = data.obank().get(retop).map(|o| o.get_addr().clone()) else { continue };
        let op = data.new_op(2, at);
        data.op_set_opcode_code(op, OpCode::CPUI_SUBPIECE);
        let Ok(high) = data.new_varnode_out(hi_size, &hi, op) else { continue };
        if let Some(v) = data.vbank_mut().get_mut(high) {
            v.set_write_mask();
        }
        let shift = data.new_constant(4, lo_size as u64);
        let _ = data.op_set_input(op, whole, 0);
        let _ = data.op_set_input(op, shift, 1);
        data.op_insert_before(op, retop);
        let _ = data.op_set_input(retop, high, 1);
    }
    true
}

/// Can [`strip_uncomputed_return_piece`] find the first register's value in
/// `whole`: the `PIECE` itself, the zero-extension of a `PIECE` whose low half
/// is the `lo_size`-byte second register, `ZEXT(x) << lo`, or a literal?
fn late_strippable(data: &Funcdata, whole: VarnodeId, lo_size: i32) -> bool {
    let def = |vn: VarnodeId| data.vbank().get(vn).and_then(|v| v.get_def()).and_then(|d| data.obank().get(d));
    let Some(op) = def(whole) else { return false };
    let piece_over_low = |x: Option<VarnodeId>| {
        x.and_then(def).is_some_and(|p| {
            p.code() == OpCode::CPUI_PIECE
                && p.num_input() == 2
                && p.get_in(1).and_then(|l| data.vbank().get(l)).is_some_and(|l| l.get_size() == lo_size)
        })
    };
    match op.code() {
        OpCode::CPUI_PIECE => op.num_input() == 2,
        OpCode::CPUI_INT_ZEXT => piece_over_low(op.get_in(0)),
        _ => shifted_high_half(data, whole).is_some() || literal_pair(data, whole).is_some(),
    }
}

/// How many operations deep [`low_bits_zero`] looks.
const ZERO_DEPTH: u32 = 6;

/// Are the low `bits` bits of `vn` zero whatever its inputs hold -- a
/// constant, or shifts, masks and concatenations that put zeros there?
/// Looks at most `depth` operations deep.
fn low_bits_zero(data: &Funcdata, vn: VarnodeId, bits: u32, depth: u32) -> bool {
    let Some(v) = data.vbank().get(vn) else { return false };
    let bits = bits.min(8 * v.get_size() as u32);
    if bits == 0 {
        return true;
    }
    if v.is_constant() {
        let mask = if bits >= 64 { u64::MAX } else { (1u64 << bits) - 1 };
        return v.get_offset() & mask == 0;
    }
    if depth == 0 {
        return false;
    }
    let Some(op) = v.get_def().and_then(|d| data.obank().get(d)) else { return false };
    let zero = |i: i32, b: u32| op.get_in(i).is_some_and(|x| low_bits_zero(data, x, b, depth - 1));
    let size = |i: i32| op.get_in(i).and_then(|x| data.vbank().get(x)).map_or(0, |x| 8 * x.get_size() as u32);
    match op.code() {
        OpCode::CPUI_COPY => zero(0, bits),
        OpCode::CPUI_INT_LEFT => {
            let shift = op.get_in(1).and_then(|c| data.vbank().get(c)).filter(|c| c.is_constant()).map(|c| c.get_offset());
            match shift {
                Some(sh) if sh >= bits as u64 => true,
                Some(sh) => zero(0, bits - sh as u32),
                None => false,
            }
        }
        OpCode::CPUI_INT_AND | OpCode::CPUI_INT_MULT => zero(0, bits) || zero(1, bits),
        OpCode::CPUI_INT_OR | OpCode::CPUI_INT_XOR | OpCode::CPUI_INT_ADD => zero(0, bits) && zero(1, bits),
        OpCode::CPUI_PIECE => {
            let low = size(1);
            if bits <= low {
                zero(1, bits)
            } else {
                zero(1, low) && zero(0, bits - low)
            }
        }
        OpCode::CPUI_INT_ZEXT | OpCode::CPUI_INT_SEXT => zero(0, bits.min(size(0))),
        OpCode::CPUI_MULTIEQUAL => (0..op.num_input()).all(|i| zero(i, bits)),
        _ => false,
    }
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

/// Is `copy` one register's move in a register-window instruction: a COPY of
/// one register into another, at a machine instruction that copies every
/// general-purpose register the prototype model passes arguments in, either
/// out of them (SPARC's `save` moves `%o0`-`%o5` into `%i0`-`%i5`) or back
/// into them (`restore`)? The register copied may already be a heritage
/// temporary, `PIECE`s of the pieces of a register the function also
/// accesses in parts (`stb %i1` reads its low byte), and the far end of the
/// instruction's other copies may be one too. `restore`'s own destination write (`restore %g0,1,%o1` is
/// `tmp = 0 + 1; <window>; %o1 = tmp`) sits at the same instruction but copies
/// the temporary the instruction computed: the compiler chose that register,
/// so it is not the window's. A move a compiler emits copies one
/// register (`mov r1,r4`) or a pair (AVR's `movw`), so it never answers
/// `true`; neither does a model with fewer than three argument registers,
/// where "every" says too little.
fn moves_register_window(data: &Funcdata, copy: OpId) -> bool {
    let is_register = |a: &Address| a.get_space().is_some_and(|sp| sp.get_type() == spacetype::IPTR_PROCESSOR);
    let Some(op) = data.obank().get(copy) else { return false };
    fn reassembled(data: &Funcdata, src: &crate::varnode::Varnode, depth: u32) -> bool {
        let is_register = |a: &Address| a.get_space().is_some_and(|sp| sp.get_type() == spacetype::IPTR_PROCESSOR);
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
        let (Some(out), Some(src)) = (
            op.get_out().and_then(|x| data.vbank().get(x)),
            op.get_in(0).and_then(|x| data.vbank().get(x)),
        ) else {
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

/// Classify the value each second register in `returned` carries at its RETURN
/// (`first_slot` is the first register's slot in a RETURN),
/// through moves only: copies `window` accepts, phis, indirects, and the
/// `PIECE`s and `SUBPIECE`s heritage splits and reassembles a register with
/// where the function also accesses part of it. The walk ends at the
/// register's entry value (at `addr`), a literal
/// ([`built_from_literals`]; the walk meets them unfolded, since it runs
/// before the rule pool) or any other operation, a value.
///
/// A RETURN whose paths end only at the entry value is a leftover. A literal
/// that also reaches the first register -- unchanged, or through any
/// operation but a sign extension -- is set aside as that register's scratch:
/// `mov 2,%i1; ret; restore %g0,%i1,%o0` builds the `int` 2, and a flag ORed
/// into the `int` in `%i0` is left in `%i1`. A `long long` whose two words are
/// the same literal is no value but -1, which is kept (clang -O0 builds
/// `return -1LL` exactly that way), or 0, which reads the same either way. A
/// computed value copied into both registers is set aside only when its top
/// bit is clear ([`top_bit_clear`]: `ldub` then `restore %g0,%i1,%o0`), where
/// the same holds; otherwise it may be -1 and keeps the pair, which reads right
/// as either type (clang -O0 reloads `c ? -1 : 0` into `%i1` and copies it into
/// `%o0`). A RETURN all of whose paths are set aside is a leftover; otherwise
/// the rest decide, so a `(0, 0)` path beside a computed pair keeps the pair.
///
/// The remaining values are followed forward, through everything computed
/// from them: when they reach only RETURNs they are deliberate, since a
/// compiler keeps no value nobody reads; when they reach a store, a call, a
/// branch or a load's address the function used them for something else on
/// the way, and literals are scratch the window happens to hand back (a
/// leftover), a computed value [`HandedValue::Computed`]. Heritage has placed
/// phis nothing reads yet; they reach nothing. Either walk past
/// [`MAX_NODES`] Varnodes gives up on a value used elsewhere.
fn window_values(
    data: &Funcdata,
    returned: &[VarnodeId],
    addr: &Address,
    first_slot: i32,
    window: &mut dyn FnMut(OpId) -> bool,
) -> Vec<HandedValue> {
    use std::collections::BTreeSet;
    struct Walk {
        seen: BTreeSet<VarnodeId>,
        ends: Vec<VarnodeId>,
        literal_only: bool,
        overflow: bool,
    }
    let mut walks: Vec<Walk> = Vec::new();
    for &vn in returned {
        let mut w = Walk { seen: BTreeSet::new(), ends: Vec::new(), literal_only: true, overflow: false };
        let mut work = vec![vn];
        while let Some(cur) = work.pop() {
            if !w.seen.insert(cur) {
                continue;
            }
            if w.seen.len() as u32 > MAX_NODES {
                w.overflow = true;
                break;
            }
            let Some(v) = data.vbank().get(cur) else { continue };
            let Some(def) = v.get_def() else {
                if !(v.is_input() && v.get_addr() == addr) {
                    w.ends.push(cur);
                    w.literal_only = false;
                }
                continue;
            };
            let Some(op) = data.obank().get(def) else { continue };
            let moves = match op.code() {
                OpCode::CPUI_COPY => window(def),
                OpCode::CPUI_INDIRECT => !op.is_indirect_creation(),
                OpCode::CPUI_MULTIEQUAL => true,
                _ => false,
            };
            if !moves && built_from_literals(data, cur, LITERAL_DEPTH) {
                w.ends.push(cur);
                continue;
            }
            let through = moves || matches!(op.code(), OpCode::CPUI_PIECE | OpCode::CPUI_SUBPIECE);
            if !through {
                w.ends.push(cur);
                w.literal_only = false;
                continue;
            }
            let n = match op.code() {
                OpCode::CPUI_MULTIEQUAL | OpCode::CPUI_PIECE => op.num_input(),
                _ => 1,
            };
            work.extend((0..n).filter_map(|i| op.get_in(i)));
        }
        walks.push(w);
    }
    walks
        .iter()
        .map(|w| {
            if w.overflow {
                return HandedValue::Computed;
            }
            if w.ends.is_empty() {
                return HandedValue::Leftover;
            }
            let builds_first = |end: VarnodeId| {
                let mut pieces = BTreeSet::new();
                value_parts(data, end, LITERAL_DEPTH, &mut pieces);
                let copied = || pieces.iter().any(|&p| reaches_slot(data, p, first_slot, true));
                if !built_from_literals(data, end, LITERAL_DEPTH) {
                    return top_bit_clear(data, end, ZERO_DEPTH) && copied();
                }
                if literal_value(data, end, LITERAL_DEPTH).is_some_and(|(k, size)| k == ones(size)) {
                    return false;
                }
                reaches_slot(data, end, first_slot, false) || copied()
            };
            let ends: Vec<VarnodeId> = w.ends.iter().copied().filter(|&e| !builds_first(e)).collect();
            if ends.is_empty() {
                return HandedValue::Leftover;
            }
            let literal_only = w.literal_only && ends.iter().all(|&e| built_from_literals(data, e, LITERAL_DEPTH));
            let mut parts: BTreeSet<VarnodeId> = BTreeSet::new();
            for &end in &ends {
                value_parts(data, end, LITERAL_DEPTH, &mut parts);
            }
            let mut carried: BTreeSet<VarnodeId> = parts.clone();
            let mut work: Vec<VarnodeId> = parts.into_iter().collect();
            let read_elsewhere = loop {
                let Some(cur) = work.pop() else { break false };
                if carried.len() as u32 > MAX_NODES {
                    break true;
                }
                let Some(v) = data.vbank().get(cur) else { continue };
                let mut real = false;
                for d in v.descend_iter() {
                    let Some(op) = data.obank().get(d) else { continue };
                    if op.is_dead() || op.code() == OpCode::CPUI_RETURN {
                        continue;
                    }
                    let sink = matches!(op.code(), OpCode::CPUI_CALL | OpCode::CPUI_CALLIND | OpCode::CPUI_CALLOTHER)
                        || (op.code() == OpCode::CPUI_LOAD && op.get_in(1) == Some(cur));
                    match op.get_out() {
                        Some(out) if !sink => {
                            if carried.insert(out) {
                                work.push(out);
                            }
                        }
                        _ => real = true,
                    }
                }
                if real {
                    break true;
                }
            };
            match (read_elsewhere, literal_only) {
                (false, _) => HandedValue::Deliberate,
                (true, true) => HandedValue::Leftover,
                (true, false) => HandedValue::Computed,
            }
        })
        .collect()
}

/// Add `vn` and the Varnodes it is assembled from to `parts`: every
/// non-constant operand of a literal ([`built_from_literals`]), or the pieces
/// heritage split a register into where the function also reads part of it
/// (`stb %i1` reads the low byte of `%i1`, so the four bytes are `PIECE`d back
/// together from pieces the store also reads). Looks `depth` operations deep.
fn value_parts(data: &Funcdata, vn: VarnodeId, depth: u32, parts: &mut std::collections::BTreeSet<VarnodeId>) {
    let Some(v) = data.vbank().get(vn) else { return };
    if v.is_constant() || !parts.insert(vn) || depth == 0 {
        return;
    }
    let Some(op) = v.get_def().and_then(|d| data.obank().get(d)) else { return };
    let reshapes = matches!(op.code(), OpCode::CPUI_PIECE | OpCode::CPUI_SUBPIECE);
    if !reshapes && !built_from_literals(data, vn, depth) {
        return;
    }
    let n = if op.code() == OpCode::CPUI_SUBPIECE { 1 } else { op.num_input() };
    for x in (0..n).filter_map(|i| op.get_in(i)) {
        value_parts(data, x, depth - 1, parts);
    }
}

/// Does `vn` reach a RETURN's `slot`: unchanged, through copies, phis,
/// indirects and additions of zero (`restore %g0,%i1,%o0` is `%o0 = 0 + %i1`),
/// or with `unchanged_only` false through any operation but one that takes
/// its sign (an arithmetic shift right by every bit but the top one, the high
/// word of a sign-extended `long long`)? Visits at most [`MAX_NODES`] Varnodes.
fn reaches_slot(data: &Funcdata, vn: VarnodeId, slot: i32, unchanged_only: bool) -> bool {
    let mut seen: std::collections::BTreeSet<VarnodeId> = std::collections::BTreeSet::new();
    let mut work = vec![vn];
    while let Some(cur) = work.pop() {
        if !seen.insert(cur) || seen.len() as u32 > MAX_NODES {
            continue;
        }
        let Some(v) = data.vbank().get(cur) else { continue };
        let bits = 8 * v.get_size() as u64;
        for d in v.descend_iter() {
            let Some(op) = data.obank().get(d) else { continue };
            if op.is_dead() {
                continue;
            }
            let constant = |x: VarnodeId| data.vbank().get(x).filter(|k| k.is_constant()).map(|k| k.get_offset());
            let follow = match op.code() {
                OpCode::CPUI_RETURN => {
                    if op.get_in(slot) == Some(cur) {
                        return true;
                    }
                    false
                }
                OpCode::CPUI_COPY | OpCode::CPUI_MULTIEQUAL => true,
                OpCode::CPUI_INDIRECT => op.get_in(0) == Some(cur),
                OpCode::CPUI_INT_ADD | OpCode::CPUI_INT_OR | OpCode::CPUI_INT_XOR if unchanged_only => {
                    (0..op.num_input()).filter_map(|i| op.get_in(i)).filter(|&x| x != cur).all(|x| constant(x) == Some(0))
                }
                OpCode::CPUI_INT_SRIGHT => !unchanged_only && op.get_in(1).and_then(constant).map_or(true, |k| k + 1 != bits),
                _ => !unchanged_only,
            };
            if follow {
                work.extend(op.get_out());
            }
        }
    }
    false
}

/// Is the top bit of `vn` zero whatever its inputs hold: a literal without
/// it, a zero-extension, a mask or a logical shift right that clears it, or
/// moves of such? Looks at most `depth` operations deep.
fn top_bit_clear(data: &Funcdata, vn: VarnodeId, depth: u32) -> bool {
    let Some(v) = data.vbank().get(vn) else { return false };
    let top = 1u64 << (8 * v.get_size().clamp(1, 8) - 1);
    if v.is_constant() {
        return v.get_offset() & top == 0;
    }
    if depth == 0 {
        return false;
    }
    let Some(op) = v.get_def().and_then(|d| data.obank().get(d)) else { return false };
    let input = |i: i32| op.get_in(i).and_then(|x| data.vbank().get(x));
    match op.code() {
        OpCode::CPUI_INT_ZEXT => input(0).is_some_and(|x| x.get_size() < v.get_size()),
        OpCode::CPUI_INT_RIGHT => input(1).is_some_and(|k| k.is_constant() && k.get_offset() > 0),
        OpCode::CPUI_INT_AND => {
            (0..op.num_input()).filter_map(|i| op.get_in(i)).any(|x| top_bit_clear(data, x, depth - 1))
        }
        OpCode::CPUI_COPY | OpCode::CPUI_MULTIEQUAL => {
            (0..op.num_input()).filter_map(|i| op.get_in(i)).all(|x| top_bit_clear(data, x, depth - 1))
        }
        OpCode::CPUI_INDIRECT => op.get_in(0).is_some_and(|x| top_bit_clear(data, x, depth - 1)),
        _ => false,
    }
}

/// The value and size of `vn` when it is a constant or a chain of copies of
/// one, at most `depth` copies deep.
fn literal_value(data: &Funcdata, vn: VarnodeId, depth: u32) -> Option<(u64, i32)> {
    let v = data.vbank().get(vn)?;
    if v.is_constant() {
        return Some((v.get_offset(), v.get_size()));
    }
    let op = data.obank().get(v.get_def()?)?;
    if depth == 0 || op.code() != OpCode::CPUI_COPY {
        return None;
    }
    literal_value(data, op.get_in(0)?, depth - 1).map(|(k, _)| (k, v.get_size()))
}

/// Every bit of a `size`-byte value set.
fn ones(size: i32) -> u64 {
    if size >= 8 {
        u64::MAX
    } else {
        (1u64 << (8 * size)) - 1
    }
}

/// How many operations deep [`built_from_literals`] looks: SPARC builds a
/// 32-bit constant in two (`sethi` then `or`).
const LITERAL_DEPTH: u32 = 4;

/// Is `vn` a constant, or computed by integer arithmetic and logic from
/// constants alone within `depth` operations?
fn built_from_literals(data: &Funcdata, vn: VarnodeId, depth: u32) -> bool {
    let Some(v) = data.vbank().get(vn) else { return false };
    if v.is_constant() {
        return true;
    }
    if depth == 0 {
        return false;
    }
    let Some(op) = v.get_def().and_then(|d| data.obank().get(d)) else { return false };
    let pure = matches!(
        op.code(),
        OpCode::CPUI_COPY
            | OpCode::CPUI_INT_ADD
            | OpCode::CPUI_INT_SUB
            | OpCode::CPUI_INT_OR
            | OpCode::CPUI_INT_XOR
            | OpCode::CPUI_INT_AND
            | OpCode::CPUI_INT_LEFT
            | OpCode::CPUI_INT_RIGHT
            | OpCode::CPUI_INT_ZEXT
            | OpCode::CPUI_INT_SEXT
            | OpCode::CPUI_INT_NEGATE
            | OpCode::CPUI_INT_2COMP
            | OpCode::CPUI_PIECE
            | OpCode::CPUI_SUBPIECE
    );
    pure && op.num_input() > 0 && (0..op.num_input()).all(|i| op.get_in(i).is_some_and(|x| built_from_literals(data, x, depth - 1)))
}

/// Does `whole` sit in two of the model's output registers, the FIRST of them
/// holding its most significant half? That is the order the output rule joined
/// them in -- a big-endian ABI's r3:r4, v0:v1 or o0:o1, and AVR's R25:R24
/// (`reversesignif`) -- read back from where the halves are stored. `false` for
/// one register, storage the model does not list, or a pair joined low half
/// first (x86-64's RAX:RDX).
pub(crate) fn first_register_holds_high(data: &Funcdata, whole: VarnodeId) -> bool {
    pair_halves(data, whole)
        .is_some_and(|(hi, hi_size, lo, lo_size)| data.get_func_proto().output_holds_high_first(&hi, hi_size, &lo, lo_size))
}

/// The storage of `whole`'s most and least significant halves -- two pieces of
/// a join, or the two halves of one register -- with their sizes.
fn pair_halves(data: &Funcdata, whole: VarnodeId) -> Option<(Address, i32, Address, i32)> {
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
#[path = "kuna_returnuncomputed/tests.rs"]
mod tests;
