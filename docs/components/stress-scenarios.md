# Synthetic stress scenarios

`evals.stress.scenarios.generate_scenario(profile, length)` creates an isolated,
deterministic conversation about the fictional Harbor scheduling project. It
accepts `growing`, `pivot`, or `contradiction` and lengths `1`, `3`, `6`, `10`,
or `20`. Unsupported profiles and lengths raise `ValueError`. Shorter cases
are prefixes of longer cases for the same profile.

The returned `Scenario` has a `profile`, an ordered tuple of `turns` (each with
one-based `number` and `message`), and a `fact_tracker` mapping each turn number
to a literal recall snippet in that turn's message. The turn index identifies
its origin separately; the snippet does not require an artificial turn label
in memory. For example:

```python
from evals.stress.scenarios import generate_scenario

case = generate_scenario('pivot', 6)
assert case.fact_tracker[5] in case.turns[4].message
```

The growing project adds requirements and tracks its original `project_name`,
`Harbor`, from turn 1 through turn 20 without repeating its name in later
messages. The pivot changes the technology stack
from React to Flutter at turn 5. The contradiction profile states
`presupuesto 30k€` at turn 3 and `presupuesto 80k€` for the same total project
budget at turn 8. Prefixes shorter than those turns omit the later events.

The tracker records historical assertions, **not** resolved current requirements:
a later conflicting fact does not erase an earlier fact. A recall measurement
can check a fact only in a later turn; retaining the earlier assertion is not a
claim that it is still the current requirement. Matching is literal and
case-insensitive: a paraphrase or changed formatting may not match, even if it
means the same thing. Tracking `Harbor` makes a later `project_name` value
eligible for that check; these fixtures alone do not prove actual retention.
They do not run a provider, inspect sessions, connect caches, or implement a
memory metric.
