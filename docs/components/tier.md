# Audience tier resolver

`app.memory.tier` exports `Tier` and the pure, keyword-only
`resolve_tier(transcript=..., metadata=..., override=None)`. It returns a
`(Tier, reason)` tuple without changing metadata or logging the input. The
metadata is the previously available `ProjectMetadata`; this resolver does not
extract team size from the current transcript. The resolver is independent of
session orchestration and prompt rendering.

The first matching rule wins:

1. An explicit `Tier` override returns that tier and `explicit_override`.
2. NDA language in the transcript or an individual agreed-scope entry returns
   `executive` / `nda_detected`.
3. Regulatory language in the transcript, an individual agreed-scope entry, or
   an individual mentioned-technology entry returns `executive` /
   `regulatory_context`.
4. At least two distinct technical keywords in the transcript return `developer`
   / `technical_audience`. Repeating one keyword, even with different case, does
   not count twice.
5. A previous `assumed_team_size` of one or two returns `pm` / `low_budget_pm`.
6. Otherwise the result is `default` / `default`.

Matches are case-insensitive and word-bounded. NDA terms are `nda`,
`non-disclosure` (hyphen or space optional), `confidential`, `under embargo`,
and `legal hold`. Regulatory terms are `hipaa`, `gdpr`, `sox`, `pci-dss`
(hyphen or space optional), `fda`, `iso-27001` (hyphen or space optional), and
`ccpa`. Technical terms are `docker`, `kubernetes`, `k8s`, `microservice`,
`microservices`, `terraform`, `iac`, `helm`, `grpc`, `graphql`, `kafka`,
`airflow`, `spark`, and `rabbitmq`. List entries are matched individually, not
joined into new phrases.

Audience controls presentation only; it is **not** an authorization or privacy
classification and must not be used to decide access or disclosure.
