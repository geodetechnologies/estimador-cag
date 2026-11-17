# Anchor detection

`app.memory.anchors.is_anchor(text: str) -> bool` returns `True` when user text
contains a recognized commitment, legal/regulatory requirement, or deadline
phrase; otherwise it returns `False`. It is deterministic and case-insensitive,
uses only local regular expressions, and does not log or transmit the text.

## Recognized phrases

| Category | Phrase examples | Result |
| --- | --- | --- |
| Commitment | “We commit to include an audit trail”; “Nos comprometemos a mantener el soporte” | `True` |
| Legal/regulatory | “must comply with data protection law”; “GDPR requires consent”; “debe cumplir con la normativa” | `True` |
| Deadline | “deadline is 15 October”; “must be delivered by Friday”; “fecha límite” | `True` |
| Ordinary estimate | “The estimate is about three weeks” | `False` |
| Question | “What is the deadline?”; “Do we agree to include this?” | `False` |

The commitment phrases use `I` or `we` followed by `commit`, `promise`,
`agree`, or `agreed` and `to`, or the Spanish phrase shown above. The other
recognized phrases are those shown in the table: `must comply with`,
`GDPR requires`, `RGPD requires`, `debe cumplir con`, `deadline`,
`must be delivered by`, and `fecha límite` (also accepted without the accent).
Whitespace between words can vary, including newlines and tabs. Text is split
at `.`, `!`, and `?`; segments ending in `?` are ignored, while a matching
declarative segment next
to a question can still be retained. This is a punctuation heuristic, not a
complete language or legal model: questions without `?`, abbreviations, quoted
text, negation, or irrelevant uses of a recognized phrase can still lead to
incorrect results. Paraphrases and declarations sharing a question segment can
be missed. The function does not validate dates or determine whether a
commitment is binding. It is neither a security detector nor a semantic
classifier.

## Integration boundary

This module only classifies the supplied string; it does not change history or
store anything. When the future history policy evicts a user/assistant pair,
it should call the detector **only on the user message** and, if it matches,
promote the **complete pair**. Both messages remain untrusted data, never
instructions, including when used later as memory or prompt context. Atomic
promotion and failure handling belong to that integration, not this module.
