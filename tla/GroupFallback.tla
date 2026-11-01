--------------------------- MODULE GroupFallback ---------------------------
EXTENDS Naturals, Sequences

CONSTANTS MaxCalls, SkipUnsupportedFlush
VARIABLES seen, forwarded, pending, passthrough, unsupported, ended, custom
vars == <<seen, forwarded, pending, passthrough, unsupported, ended, custom>>
Ops == {"rank0", "rank1", "unsupported"}
Pair(s) == IF Len(s) = 2 THEN {s[1], s[2]} = {"rank0", "rank1"} ELSE FALSE

Init == /\ seen = <<>> /\ forwarded = <<>> /\ pending = <<>>
        /\ passthrough = FALSE /\ unsupported = FALSE
        /\ ended = FALSE /\ custom = FALSE

Receive(op) ==
  /\ ~ended /\ Len(seen) < MaxCalls /\ op \in Ops
  /\ seen' = Append(seen, op)
  /\ unsupported' = (unsupported \/ op = "unsupported")
  /\ IF op = "unsupported" /\ SkipUnsupportedFlush
        THEN /\ forwarded' = Append(forwarded, op)
             /\ UNCHANGED <<pending, passthrough>>
        ELSE IF passthrough \/ op = "unsupported" \/ Len(pending) = 2
          THEN /\ forwarded' = Append(forwarded \o pending, op)
               /\ pending' = <<>> /\ passthrough' = TRUE
          ELSE /\ pending' = Append(pending, op)
               /\ UNCHANGED <<forwarded, passthrough>>
  /\ UNCHANGED <<ended, custom>>

End == /\ ~ended /\ ended' = TRUE
       /\ custom' = (~passthrough /\ Pair(pending))
       /\ forwarded' = IF ~passthrough /\ Pair(pending)
                         THEN forwarded ELSE forwarded \o pending
       /\ pending' = <<>>
       /\ UNCHANGED <<seen, passthrough, unsupported>>

Next == (\E op \in Ops : Receive(op)) \/ End
Spec == Init /\ [][Next]_vars
OrderedPrefix == /\ Len(forwarded) <= Len(seen)
                 /\ forwarded = SubSeq(seen, 1, Len(forwarded))
PendingBound == Len(pending) <= 2
BufferedOrder == ~ended => forwarded \o pending = seen
Complete == ended => IF custom
              THEN /\ Pair(seen) /\ ~unsupported /\ forwarded = <<>>
              ELSE forwarded = seen
=============================================================================
