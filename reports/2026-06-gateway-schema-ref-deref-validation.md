# Gateway JSON-Schema `$ref`/`$defs` dereferencing — runtime validation (2026-06)

Validation of the devshard gateway's JSON-Schema `$ref`/`$defs` dereferencing path
([gonka-ai/gonka#1310](https://github.com/gonka-ai/gonka/pull/1310)). The gateway now accepts
client schemas that use local `$ref`/`$defs`, dereferences them inline, strips the definition
containers, and forwards a **flat** schema to vLLM — so vLLM never sees a `$ref`. This report
checks the crash-safety of that path end-to-end.

## Summary

We ran the dereferencing code and the full parameter validators **unmodified, at production
limits**, over 33 adversarial `$ref`/`$defs` inputs, and replayed every gateway-accepted (flat)
schema against **live vLLM 0.20.0** on a MiniMax-M2.7 engine, cross-checked on a Kimi-K2.6
engine. Grammar compilation was forced (via `response_format` and via a pinned `tool_choice`) so
xgrammar actually compiled each forwarded schema.

**No vLLM crash on any input.** Engine `/health` stayed green before and after every batch — no
restarts, no `/health` flaps, no service-state transitions. The dangerous classes (recursion,
SSRF, oversize, CVE-class `pattern`/`type`) are all rejected at the gateway before reaching vLLM;
the schemas that pass either compile cleanly (200) or are cleanly rejected by vLLM (400) — none
crash the engine. The crash-safety the change is designed for holds end-to-end.

## Method

The code is pure Go in `devshard/cmd/devshardctl/paramvalidators`. We ran it through a small probe
harness at the production limits per surface:

| Surface | MaxDepth | MaxNodes | MaxSize | MaxBranch | MaxEnum | MaxPatternLen |
|---|---|---|---|---|---|---|
| `response_format.json_schema` | 16 | 128 | 16 KiB | 16 | 256 | 512 |
| `structured_outputs.json` | 16 | 128 | 16 KiB | 16 | 256 | 512 |
| `tools[].function.parameters` | 16 | 256 | 16 KiB | 16 | 256 | 512 |

For each accepted input we captured the exact flat schema the gateway forwards, then sent it to
live vLLM inside a real chat-completions request, forcing grammar compilation. Both halves are
real: the dereferencing is the actual gateway code, the downstream is the production engine +
xgrammar.

## 1. Defenses verified working (gateway rejects → never reaches vLLM)

| Adversarial input | Gateway result | Sentinel |
|---|---|---|
| direct self-cycle `A→A` | reject | `ErrSchemaRef` (recursive) |
| indirect cycle `A→B→A` | reject | `ErrSchemaRef` (recursive) |
| cycle via root `$ref:"#"` | reject | `ErrSchemaRef` (recursive) |
| external `http://` ref (SSRF) | reject | `ErrSchemaRef` (only local) |
| external `file://` ref | reject | `ErrSchemaRef` (only local) |
| relative `schema.json#/X` | reject | `ErrSchemaRef` (only local) |
| unresolved pointer `#/$defs/Missing` | reject | `ErrSchemaRef` (unresolved) |
| DAG-bomb (exponential expansion) over budget | reject | `ErrSchemaNodes` |
| over-budget `anyOf` breadth / `enum` size | reject | `ErrSchemaNodes` / `ErrSchemaEnum` |
| unclosed regex `pattern:"("` inside a `$def` | reject | `ErrSchemaPattern` |
| invalid `type:"something"` inside a `$def` | reject | `ErrSchemaType` |
| `$ref` to a non-object **with** sibling keys | reject | `ErrSchemaRef` |
| empty / whitespace-only `$ref` | reject | `ErrSchemaRef` |

Two things worth calling out as well-engineered:

- The CVE-2025-48944 `pattern`/`type` checks fire **after** dereferencing, so a bad pattern or
  type hidden inside a `$def` is still caught once it is inlined.
- The node-budget counter operates on the **expanded** tree, so a DAG that is small as written
  but explodes when inlined is correctly rejected.

## 2. Forwarded schemas on live vLLM (gateway accepts → sent to engine)

Engine healthy before and after.

| Case | vLLM |
|---|---|
| pointer to array index `#/$defs/L/1` | 200 |
| JSON-pointer `~0` / `~1` escaping in keys | 200 |
| sibling override (`$ref` + extra keys) | 200 |
| `$ref` to a non-object scalar (no siblings) | **400** `schema must be an object or boolean` |
| `$ref` inside an `enum` literal | 200 |
| one `$def` inlined 8× | 200 |
| 10-level-deep chained refs | 200 |

The single 400 is a **clean vLLM rejection, not a crash** (see Observation 1). Behaviour was
identical on the Kimi-K2.6 engine for the cross-checked cases.

## 3. All three call-sites + budget ceiling

The dereferencing is wired into three surfaces. All three behave identically, and the **budget
ceiling values themselves are safe** for xgrammar (not just enforced).

| Test | Surface | vLLM | Notes |
|---|---|---|---|
| 250 properties (wide, ~251 nodes) | tools | 200 | compiles at the ceiling, sub-second |
| 125 properties | response_format | 200 | — |
| 16-level deep nesting | tools | 200 | at depth ceiling |
| 16-arm `anyOf` (1 level) | tools | 200 | at branch ceiling |
| 256-entry `enum` | tools | 200 | at enum ceiling |
| `$ref` ×2 (shared sub-schema) | structured_outputs | 200 | third call-site deref works |
| 8 tools, pinned `tool_choice` | tools | 200 | real `tool_call`; aggregate grammar compiles |

Every worst-case legal schema at a ceiling compiled in well under one second with
`finish_reason=length` (grammar actually applied), engine healthy throughout. Multi-tool
aggregate grammars compile cleanly too.

## Observations and suggestions

These are minor; none are crashes, and none block the change.

1. **`$ref` to a non-object scalar** (`$defs:{"S":"x"}` + `{"$ref":"#/$defs/S"}`) is forwarded
   verbatim and produces a schema vLLM always rejects with a clean 400. Suggest rejecting at the
   gateway with a targeted error — mirroring the existing "cannot merge siblings into non-object"
   path that already handles the with-sibling case — so clients get a clear gateway error instead
   of a downstream one. (`paramvalidators/schema_refs.go`)

2. **`$ref` inside `enum`/`const` literals** is treated as literal data: not dereferenced, and
   not caught by the post-deref `$ref` ban. vLLM treats it as a literal value (harmless). Worth a
   one-line doc note that the "vLLM never sees `$ref`" invariant is a schema-position guarantee,
   not a literal-data guarantee.

3. **Sibling-key merge overrides the target** rather than AND-merging it. Per JSON Schema 2020-12,
   sibling keys to `$ref` apply alongside the target (effectively `allOf`); draft-07 ignores them
   entirely. The override is fine and is what the tests lock in — worth documenting the chosen
   semantics explicitly so client expectations are clear.

4. **Regex bounds could be applied symmetrically.** The gateway length/compile-checks `pattern`
   values; the same check could be extended to other regex-bearing positions
   (`patternProperties` keys, `propertyNames.pattern`) so the regex policy is uniform regardless
   of where a regex sits. We tested pathological regexes in these positions against live vLLM and
   confirmed xgrammar (Rust, linear-time) handles them without crashing — so this is a
   defense-in-depth tidy-up, not a live issue.

Suggested regression tests (all exercised here and correct): external-ref reject, JSON-pointer
escaping and array-index resolution, unresolved pointer, `$ref` to a non-object, cycle via root
`#`, `$ref` survival inside `enum`/`const`, and CVE-class `pattern`/`type` inside a `$def`.

## Conclusion

The `$ref`/`$defs` dereferencing path is crash-safe: across 33 adversarial inputs through the real
validator code and ~20 forwarded schemas on two live engines — covering all three call-sites and
every budget ceiling — vLLM never crashed. The dangerous schema classes are rejected at the
gateway; the safe ones compile cleanly. The observations above are minor follow-ups.

---

_Source: Kaitaku.ai — https://github.com/kaitakuai/experiments_
