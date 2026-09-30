# bejoin: plan (as built)

Strict correctness fix, no option: it changes only register-pair returns and
call outputs whose output rule joins the first register high (every big-endian
target, and AVR's gcc spec through `reversesignif`); every other target is
byte-identical.

1. `ParamActive::join_pair_order` (`p4_calls/fspec.rs`) names which of the first
   two used trials is the low half, from the flag the matched output rule sets.
2. `ActionReturnRecovery::build_return_output` builds `PIECE(hi, lo)` and the join
   address from that order; `FuncCallSpecs::build_output_from_trials` passes it to
   `kuna_rustabi::build_call_output_pair`, which assigns the SUBPIECE offsets by
   it. The call classifier keeps reasoning in register order (the veto asks about
   the second register), and `holds_scalar_pair` maps the PIECE halves back to
   register order before asking for the Rust tag.
3. `kuna_returnuncomputed`:
   - `classify_window_pair`, called after the output map is derived,
     classifies a pair joined first register high (`WindowPair`, kept on
     `Funcdata::kuna_window_pair`). The second register must be defined at every
     live RETURN by a register-window move (`moves_register_window`: a COPY of
     one register into another, the register possibly a heritage `PIECE` of its
     pieces, at an instruction copying every general-purpose argument register,
     SPARC `save`/`restore`; `restore`'s destination write copies a temporary
     and is not one). `window_values` follows `%i1` back through window copies,
     phis and indirects, then `PIECE`s and `SUBPIECE`s, to the entry value,
     literals (`built_from_literals`) and values. A literal that reaches the
     first register's slot unchanged or through any op but a sign extension
     (`reaches_slot`), unless it is all ones (`literal_value`), and a computed
     value copied unchanged there whose top bit is clear (`top_bit_clear`), are
     set aside; a RETURN all of whose paths are set aside holds a leftover. The
     rest are followed forward with the pieces they are built from
     (`value_parts`): reaching only RETURNs is *deliberate*; a literal reaching
     a store, a call, a branch or a load address is a leftover, as is the entry
     value; a used computed value is *handed back*. Any deliberate RETURN makes
     the pair `WindowPair::No`; otherwise any leftover RETURN makes it
     `Leftover`; otherwise `HandedBack`. RETURNs behind a branch literals decide
     the other way are skipped (`never_reached`, `decided`):
     SPARC's call pcode keeps one for a delay-slot `restore`.
   - `zero_leftover_low_half`, right after the join and at the start of every
     later pass of return recovery: a leftover's `PIECE` low input becomes zero
     where the high input is zero, so the join folds into a literal rather than
     `ZEXT(leftover)`.
   - the late repair (`strip_uncomputed_return_piece`, per RETURN in
     `repair_return`) reads, for any window pair, `ZEXT(x) << lo`
     (`shifted_high_half`: keep `x`) and a literal (`literal_pair`: keep a new
     constant of its high half; ActionOutputPrototype stores it in the first
     register, `window_high_storage`); a leftover's second register counts as
     leftover in the `PIECE` path. A window pair is repaired at every live
     RETURN or at none; it repoints every RETURN before removing any
     concatenation.
   - `clear_dead_varnodes` (substrate) takes a freed Varnode out of its
     HighVariable, as C++ `~Varnode` does: the late repair can leave a function
     input unread after merging.
   - `narrow_window_pair`, on every pass of return recovery after the join:
     when some RETURN has a shape the late repair does not read
     (`late_strippable`) and the second register is a leftover or the low half is
     zero at every RETURN (`low_bits_zero`), each RETURN gets
     `SUBPIECE(whole, lo)` in the first register.
   - `first_register_holds_high`: the order read back from storage (the high
     half in the earlier model output entry,
     `ParamListStandard::holds_high_first`); the late repair uses it to look
     through `ZEXT(PIECE(x, lo))` and keep `x`, and for the first-in-class
     tie-break when both halves are uncomputed; `holds_scalar_pair` uses it to
     map PIECE halves back to register order.
4. `kuna_rustabi::pair_join_address`: `constructJoinAddress`, except that a
   contiguous pair whose parent register is global storage (AVR's `R25R24`) gets
   a join record, so the value is not the global variable its bytes alias. Used
   by both the return and the call-output pair.
5. Tests: `tests/stages/kuna-bejoin.xml` (PowerPC, 5 asserts, 0/5 before),
   `tests/stages/kuna-bejoin-sparc.xml` (SPARC, 9 asserts: 2/9 on main, 4/9
   with the second version), `tests/stages/kuna-bejoin-sparc64.xml` (SPARC
   functions that return their low word on purpose, 10 asserts: 4/10 on main,
   1/10 with the version before it), `tests/stages/kuna-bejoin-avr.xml` (AVR, 5
   asserts, 2/5 on main, 1/5 with the first version); compiled round trips
   `a_big_endian_register_pair_round_trips_through_the_printed_c` (six
   `bejoin_*.o`), `an_argument_carried_across_a_call_into_the_low_word_is_returned`
   (`bejoin_carry_arm32_{be,le}.o`),
   `an_avr_register_pair_joins_high_byte_first_as_a_value` (`bejoin_avr.bin`) and
   `a_register_window_leftover_is_not_the_low_word`
   (`bejoin_window_sparc32_O{0,2}.o`) and
   `a_low_word_a_sparc_function_returns_on_purpose_is_part_of_the_value`
   (`bejoin_window64_sparc32_O{0,2}.o`); unit tests for the call-pair order, the
   window walk (leftover, deliberate, used), literals built in steps, a zero low
   word, the shapes the late repair reads, decided conditions and never-reached
   RETURNs.
