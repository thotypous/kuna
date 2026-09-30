# bejoin: a two-register value is joined in the order its rule consumed the registers

## The defect

A 32-bit ABI returns a 64-bit value in two registers. Every big-endian ABI kuna
supports puts the HIGH word in the first register, because a register pair holds
a wide value as a load from memory would, lower address first:

| ABI | pair | high word | evidence |
|---|---|---|---|
| PowerPC SVR4 | r3:r4 | r3 | clang `-O2 wide_mul`: `mulhw 5,4,3; mullw 4,4,3; mr 3,5` |
| MIPS o32 (BE) | $2:$3 | $2 | `mult $5,$4; mfhi $2; mflo $3` |
| SPARC | %o0:%o1 | %o0 | `smul %i1,%i0,%i1; rd %y,%i0` (the product's high word from `%y`) |
| ARM AAPCS (BE) | r0:r1 | r0 | `smull r2,r3,r1,r0; mov r0,r3; mov r1,r2` |
| AArch64 AAPCS64 (BE) | x0:x1 (`__int128`) | x0 | `smulh x8,x1,x0; mul x1,x1,x0; mov x0,x8` |

The cspecs say so too: a `<join/>` output rule consumes the most significant piece
first on a big-endian register space (`MultiSlotAssign::consumeMostSig`), and the
rule records it on the trial container (`ParamActive::isJoinReverse`).

The flag belongs to the rule, not to the target: `reversesignif` flips it. AVR's
gcc spec (`avr8gcc.cspec`) is little-endian, lists `R25` first and joins with
`reversesignif="true"`, so an AVR `int` comes back with the high byte in `R25`,
the first register, and sets the flag too. AVR is the only shipped spec that
does.

Return recovery ignored that flag. `ActionReturnRecovery::buildReturnOutput`
(upstream too) and kuna's call-output pair (`kuna_rustabi::build_call_output_pair`,
the completion of `FuncCallSpecs::buildOutputFromTrials`, which upstream does get
right) both took trial 0, the first register, as the LOW word. On main, clang
`-O2 long long wide_mul(int a,int b){return (long long)a*b;}` printed
`return CONCAT44(a1 * a0,(int)((unsigned long)((long)a1 * a0) >> 0x20));` on PPC32,
MIPS-BE, SPARC and ARM-BE: the two halves swapped. AVR `int negate(int a)`
printed `return CONCAT11(a1,a0);`, R24 as the high byte.

## AVR: the right order lands on a global

AVR maps its register file into data memory (`REGISTER_SPACE "mem"`, a global
range), so with the order right the two contiguous registers name `R25R24`, a
global variable whose bytes are the argument registers themselves. Return
recovery built the whole there (`constructJoinAddress` returns the parent
register for a contiguous pair), address-tied merging grouped it with `a0`/`a1`,
`markInternalCopies` made the PIECE non-printing, and `negate` printed
`return R25R24;`, a register the printed C never assigns. `pair_join_address`
keeps the join record when the parent register is global storage, so the pair is
a value (`return CONCAT11(a0,a1);`). Register-space parents are unaffected.

## What the correct order exposed

`kuna_returnuncomputed` drops a pair's passive half. On SPARC, `restore` copies
every in-register back to its out-register, so `%o1` reaches the RETURN holding
whatever the function last left in `%i1` (the incoming second argument when it
never touched it) and passes ancestor realism (GH-6990). With the wrong order that
leftover was the high half; the repair saw the `PIECE` and kept `%o0`, or the rule
pool folded `PIECE(0, x)` into `ZEXT(x)` and the return narrowed to `%o0`. With
the ABI order the value is the high half, and the rule pool folds a zero-extended
or zero half out of the join before the repair runs (`PIECE(ZEXT(x),y)` ->
`ZEXT(PIECE(x,y))`, `PIECE(0,y)` -> `ZEXT(y)`, `PIECE(x,0)` -> `ZEXT(x) << 32`).
The first version of the change printed `u4_equals` as `return CONCAT14(a0 ==
a1,a1)` and `pcode_AccessGlobal` as `return a1`; the second fixed those for the
untouched argument, but still turned functions main printed right into wide
returns of their value in the high word: clang -O0 zeroes `%i1` to store a byte
(`mov %g0,%i1`; Lua's `createstrobj`, `scanformat`), an unrelocated `sethi
%hi(sym),%i1` leaves a zero there (`str_dump`), a loop exits on the zero it last
loaded into `%i1` (SQLite's `closePendingFds`, `sqlite3_database_file_object`), and
at every `call` SPARC's pcode keeps a RETURN for a delay-slot `restore` that
`didrestore = 0` never reaches, where `%o1` is the call's argument. SQLite's
`sqlite3_libversion` printed `return 0x4000000200000000;` for `return
0x40000002;`, and `zero_after(a,b){ext2(b,a);return 0;}` printed `return a1;`.

The fix therefore also:

- classifies, in return recovery, the second register of a pair joined first
  register high (`classify_window_pair`). At every live RETURN it must be the
  output of a register-window move: a copy of one register into another (the
  register possibly a heritage temporary reassembled from its pieces, where the
  function also reads its low byte) at an instruction that copies every
  general-purpose argument register, out of them or back into them (`save`,
  `restore`). `restore`'s own destination write is not one: `restore
  %g0,1,%o1` is `tmp = 0 + 1; <window copies>; %o1 = tmp`, so a function that
  writes `%o1` that way returns an ordinary pair. Behind the window move, the
  value in `%i1` is followed back through the window's copies, phis, indirects
  and heritage pieces to the entry value, a literal or another value
  (`window_values`), then forward. A literal or value that, with everything
  computed from it, reaches only RETURNs was put there to be returned (`mov
  10,%i1; ret; restore %g0,%g0,%o0`, `mix2`'s product), and the pair is an
  ordinary one. The entry value is a leftover; so is a literal that reaches a
  store, a call, a branch or a load's address (`mov %g0,%i1` before a byte
  store, an unrelocated `sethi` address half). A literal that also reaches
  `%o0`, unchanged or through an operation other than a sign extension, is set
  aside as that register's scratch (`mov 2,%i1; ret; restore %g0,%i1,%o0`
  builds the `int` 2; Lua's `lua_iscfunction` ORs a flag left in `%i1` into
  the `int` in `%i0`), unless it is -1: clang -O0 builds `return -1LL` exactly
  that way, and both readings agree. A computed value copied into both
  registers is set aside only when its top bit is clear (SQLite's `ldub` then
  `restore %g0,%i1,%o0`); one that may be -1 keeps the pair, which reads right
  as either type (clang -O0 reloads `c ? -1 : 0` into `%i1` and copies it into
  `%o0`). A RETURN all of whose paths are set aside is a leftover; otherwise
  the rest decide, so Lua's `luaV_mod`, which hands back `(0, 0)` on one path
  and its 64-bit remainder on another, keeps the pair. A computed value used
  on the way is *handed back*. A pair returned on purpose at any RETURN is an
  ordinary pair; otherwise a leftover at some RETURN makes it a leftover (an
  `int` function with the untouched argument in `%i1` on its error path and a
  loop counter on the other, `sum_or`); a pair with a used computed value at
  every RETURN is handed back. A move a compiler chose copies one register, so ARM
  big-endian's `mov r4,r1; bl ext; mov r0,#0; mov r1,r4` (the second argument
  carried across a call into the low word of a `long long`) keeps its pair.
  RETURNs reached only along branches literals decide the other way (the call's
  `if (didrestore == 0) goto next; return [o7];`) are skipped. An earlier
  version counted any copy at the `restore` instruction as the window's and
  every literal in `%i1` as scratch: `long long k_one(void){ext();return 1;}`
  (`restore %g0,1,%o1`) printed `return 0`, `neg_one` (`restore %g0,%i0,%o1`)
  the 32-bit `0xffffffff` where main was right, and `sel_const` (`mov 10,%i1`)
  `return 0` without its argument;
- builds the pair either way. The late repair keeps a leftover pair's `%o0`
  from the `PIECE` (a literal `%o1` counts as leftover there). Next to a zero
  `%o0` the leftover would decide the fold (`PIECE(0, a1)` -> `ZEXT(a1)`, the
  argument), so it is replaced with zero in that RETURN's join, right after the
  join and again at the start of every later pass, since `%o0` can become zero
  on one path only after conditional constant propagation
  (`zero_leftover_low_half`). For any window pair the late repair also reads the
  two shapes a zero low half folds into: `ZEXT(x) << 32` (`shifted_high_half`,
  also where conditional constant propagation shows a handed-back register zero,
  the zero a loop exits on) and a literal (its high half becomes a new constant
  stored in `%o0`). Only a pair folded into another shape is narrowed during
  the main loop (`SUBPIECE(whole, lo)` in `%o0`, `narrow_window_pair`). A
  window pair is repaired at every live RETURN or at none: when one RETURN keeps
  both registers, the others keep them too (the earlier version returned `a0`
  for `mix2`'s `(u64)x << 32` beside the product's pair). The repair repoints
  every RETURN before removing any concatenation. Removing a window pair's
  whole chain can leave `%o1`'s input unread; the dead-code pass then freed that
  input without taking it out of the merged variable that listed it, so the
  naming pass read a stale Varnode (SQLite's `replaceFunc` stopped). The dead
  Varnodes it clears now leave their variable first, as C++'s `~Varnode` does
  (`clear_dead_varnodes`; x86-64 castbench, the typesweep and every
  little-endian object are unchanged by it). Narrowed there: a
  leftover (`CONCAT44(x, a1) & 0xffffffffffff` from a masked return) or a pair
  whose low half is zero there (`(ZEXT(x) << 32) ^ k`). Three earlier versions of this round
  narrowed during the main loop more widely, or zeroed every leftover; a single
  `%o0` return present during merging becomes one variable with the first
  argument wherever kuna already prints a call's unrecovered `%o0` result as `a0`
  (all of these unrelocated `.o` calls), and SQLite printed `a0 = (struct_100
  *)0x1; return a0;` for `return 1;` (up to 192 more `a0 =` lines, 31 Lua
  functions against main's 2); zeroing every leftover typed the value through
  the fold and spelled GH-6990's `unsigned int main` as `uint4`. As built, 2
  functions print that shape and 3 stop printing it across the three corpora;
- has the late repair look through `ZEXT(PIECE(x, lo))` on a pair joined first
  register high (the argument reloaded from its stack slot at `-O0`);
- keeps the first-in-class register (the high half) when both halves are
  uncomputed, and recognizes a pair whose two registers merged into one wider
  Varnode (SPARC `%o0:%o1` at `register:0x20`, not a join).

Only the entry value is truly ambiguous: `int f(int,unsigned){ext();return 0;}`
and `unsigned long long f(int,unsigned b){ext();return b;}` are the same SPARC
bytes, and the one-register reading is taken, as main took it and as a leaf
function is read. A literal or value the function writes into `%o1` through
`restore`, or into `%i1` for the window to hand back and reads nothing else from,
is told apart by what reads it. A computed `%i1` the function also used, zero at
every exit only by the control flow (a loop pointer), is still read as scratch,
so a genuine `(u64)x << 32` built that way prints as `x`.

Against DWARF return types (the same sources built with `-g` for an ILP32
target), the printed return width on SPARC matches the source for, main /
earlier version / now: SQLite 236 / 376 / 439 of 1,188 functions (67 fixed, 4
broken against the earlier version: three keep a pair every RETURN of which
hands back a computed, used `%i1`, and one loads `%i0:%i1` with one `ldd` and
returns); zlib 39 / 50 / 53 of 107; bzip2 and zlib at -O0 and -O2 51 / 62 / 68
of 198. On Lua (1,334 functions), 31 SPARC widths improve on main and 3 regress
(`lua_load`, `str_rep`, `codepoint`: every RETURN hands back a computed, used
`%i1`, and main printed them narrow only because its per-RETURN repair
narrowed the first RETURN and kept the pair at others).

Every decision after the join reads the order from storage
(`first_register_holds_high`: the high half is in the earlier model output
entry), which is the rule's order whether the target is big-endian or uses
`reversesignif`; the classification reads the same order from the live trials.

## Not changed

Two-register parameters are never joined (`ActionParamDouble` is a no-op), so they
print as two arguments in the ABI's order, which was already right. The 68000
cspec names its pair in one join entry (`D0:D1`); its trials sort by justified
offset, which already put the low half first. A little-endian rule without
`reversesignif` never sets the flag, and no register-space pair is global, so
those targets (x86, ARM, MIPSel, PowerPC LE, and every other shipped little-endian
spec but AVR's gcc one) are byte-identical by construction and by measurement.

## Residual, pre-existing, out of scope

- SPARC's window hands back whatever `%i1` holds. When that is a value the
  function computed and also used at every RETURN, and it is not zero at every
  return, the pair still forms, as on main, printed in ABI order (`int
  scale(int *p,int k)`, which stores the product it leaves there). So does a
  value the function only used to compute `%o0` (`add %i0,%i1,%i0` leaving a
  partial sum in `%i1`): a genuine `long long` builds its high word from its low
  one the same way (a sign extension, `0x80000000 | 1`-style flags), so that is
  not told apart. That is common: clang -O0 reloads spilled values
  into `%i1` (`int same(unsigned,unsigned)` prints `CONCAT14(a0 == a1,a0)`),
  bzip2's -O0 blocksort functions return one, and so do functions that leave a
  mask or a loop counter there. On Lua's SPARC objects about 60% of functions
  print such a pair on main and here; where main put the real `%o0` value in the
  low word by accident, this change puts it in the high word, where the ABI has
  it.
- SPARC calls in kuna's `.o` images are unrelocated (`R_SPARC_WPLT30` is not
  applied), so a call's `%o0` result is not recovered and prints as the first
  argument (`g(a0,a1); v1 = a0;`), on main and here.
- Without a register window nothing tells a leftover from a low word: clang -O0
  MIPS materializes `addiu $3,$zero,0` in bzip2's `handle_compress`, the machine
  code of `return (unsigned long long)x << 32` (`hi_only`, which this change
  prints right on MIPS, PowerPC and ARM big-endian). It now prints `return
  (unsigned long)v4 << 0x20;` where the swapped order narrowed it to the `bool` by
  accident.
- A zero-extended return narrowed to its low word is typed `int`, not
  `unsigned int` (`((a + 1) << 16) | b` returned as `unsigned long long`), on both
  endiannesses; filed as #769.
- An argument kept in a callee-saved register across a call and moved back into
  its own register to be returned is taken for leftover by the late repair's
  placement test, on both endiannesses: ARM little-endian
  `(u64)b << 32 | a` after a call prints `return a0` on main and here. With the
  pair in ABI order the same test turns AArch64 big-endian
  `unsigned __int128 c(unsigned long a, unsigned long b){ext();return b;}` into
  `return 0` (main printed it swapped, `a1 << 0x40`); filed as #770. The same
  test drops a pair both of whose halves pass straight through at -O0 on PowerPC
  and ARM big-endian: `u64 mix(unsigned a,unsigned b){return ((u64)a<<32)|b;}`
  printed `CONCAT44(a1,a0)` on main (swapped) and now prints `unsigned int
  mix(unsigned int a0){return a0;}`, which is what main and this change print for
  the mirror function on PowerPC LE and ARM LE.
- AVR output is otherwise rough (registers print as globals, `R1 = 0;`, most
  functions with a call print `void`); only the pair order and the global pair
  changed here.
- A call output that uses only the SECOND register (`(unsigned)f()` on BE, the
  high word on LE) is not recognized on either endianness.
- MIPS o32 drops a returned half whose register is also read by a compare
  (`sltu`) on both endiannesses (`void add_one(void)` on MIPSel).
- PPC64 `__int128` returns recover only r3.
