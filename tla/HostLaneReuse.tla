--------------------------- MODULE HostLaneReuse ---------------------------
EXTENDS Naturals, FiniteSets

CONSTANTS Pieces, Lanes, SkipReuseWait
VARIABLES produced, consumed, host, output
vars == <<produced, consumed, host, output>>
Chunks == 1..Pieces
Lane(k) == (k - 1) % Lanes

Init == /\ produced = {} /\ consumed = {}
        /\ host = [lane \in 0..(Lanes - 1) |-> 0]
        /\ output = [k \in Chunks |-> 0]


Produce(k) ==
  /\ k \in Chunks \ produced
  /\ (k <= Lanes \/ (k - Lanes) \in produced)
  /\ (SkipReuseWait \/ k <= Lanes \/ (k - Lanes) \in consumed)
  /\ host' = [host EXCEPT ![Lane(k)] = k]
  /\ produced' = produced \cup {k}
  /\ UNCHANGED <<consumed, output>>

Consume(k) ==
  /\ k \in produced \ consumed
  /\ (k <= Lanes \/ (k - Lanes) \in consumed)
  /\ output' = [output EXCEPT ![k] = host[Lane(k)]]
  /\ consumed' = consumed \cup {k}
  /\ UNCHANGED <<produced, host>>

Next == (\E k \in Chunks : Produce(k)) \/ (\E k \in Chunks : Consume(k))
Spec == Init /\ [][Next]_vars
CorrectOutputs == \A k \in consumed : output[k] = k
ProducedBeforeConsumed == consumed \subseteq produced
=============================================================================
