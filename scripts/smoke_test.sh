#!/usr/bin/env bash
#
# Verify that a built server binary actually starts and serves.
#
# This is the check whose absence let a stale binary ship: dist/server_module
# was frozen from a commit that predated the SQLite storage backend, crashed on
# startup, and nothing noticed.
#
# Usage:
#   scripts/smoke_test.sh <path-to-server-binary> [expected-git-sha]
#
# Runs the binary against a small self-contained fixture registry (see
# scripts/make_fixture_db.py) so it needs neither example.db nor the 477 MB
# training CSV. Exits non-zero on the first failed assertion.
set -euo pipefail

BINARY="${1:?usage: smoke_test.sh <path-to-server-binary> [expected-git-sha]}"
EXPECTED_SHA="${2:-}"
PORT="${SMOKE_PORT:-8123}"
BASE="http://127.0.0.1:${PORT}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKDIR="${SMOKE_DIR:-$ROOT/.smoke}"
FIXTURE_DIR="$WORKDIR/fixture"
SERVER_PID=""

if [[ -z "${PYTHON:-}" ]]; then
  if command -v python >/dev/null 2>&1; then PYTHON=python; else PYTHON=python3; fi
fi

cleanup() {
  if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

fail() { echo "SMOKE TEST FAILED: $*" >&2; exit 1; }
pass() { echo "  ok: $*"; }

# ---------------------------------------------------------------- preconditions
echo "==> smoke test: $BINARY"
[[ -f "$BINARY" ]] || fail "binary not found: $BINARY"
[[ -x "$BINARY" ]] || fail "binary is not executable: $BINARY"
BINARY="$(cd "$(dirname "$BINARY")" && pwd)/$(basename "$BINARY")"

# ------------------------------------------------------------- fixture registry
echo "==> building fixture registry in $FIXTURE_DIR"
rm -rf "$FIXTURE_DIR"
mkdir -p "$FIXTURE_DIR"
"$PYTHON" "$ROOT/scripts/make_fixture_db.py" "$FIXTURE_DIR" >/dev/null \
  || fail "could not build fixture registry"

# ------------------------------------------------------------------ start binary
echo "==> starting server on port $PORT"
# Run with cwd set to the fixture so the binary picks up its .env, which is how
# it is deployed.
( cd "$FIXTURE_DIR" && PORT="$PORT" HOST=127.0.0.1 exec "$BINARY" ) \
  > "$WORKDIR/server.log" 2>&1 &
SERVER_PID=$!

ready=""
for _ in $(seq 1 40); do
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "--- server log ---" >&2; cat "$WORKDIR/server.log" >&2
    fail "server process exited during startup"
  fi
  if curl -fsS --max-time 2 "$BASE/health" >/dev/null 2>&1; then ready=1; break; fi
  sleep 0.5
done
[[ -n "$ready" ]] || { echo "--- server log ---" >&2; cat "$WORKDIR/server.log" >&2; fail "server never became ready on $BASE"; }
pass "server is up"

# ------------------------------------------------------------------- /health
echo "==> GET /health"
HEALTH="$(curl -fsS --max-time 5 "$BASE/health")"
echo "    $HEALTH"
read -r STATUS COUNT SHA <<<"$(printf '%s' "$HEALTH" | "$PYTHON" -c '
import json,sys
d=json.load(sys.stdin)
print(d.get("status",""), d.get("model_count",0), (d.get("build") or {}).get("git_sha",""))
')"
[[ "$STATUS" == "ok" ]] || fail "health status is '$STATUS', expected 'ok' (failures: $HEALTH)"
[[ "$COUNT" -ge 1 ]] || fail "server loaded $COUNT models, expected at least 1"
pass "health ok with $COUNT model(s)"

# --------------------------------------------------------------- provenance
# A binary built from a different commit than the one under test is the exact
# failure mode this script exists to catch.
if [[ -n "$EXPECTED_SHA" ]]; then
  if [[ -z "$SHA" || "$SHA" == "unknown" ]]; then
    fail "binary carries no provenance stamp; rebuild with tools/stamp_build.py"
  fi
  [[ "${SHA%-dirty}" == "${EXPECTED_SHA%-dirty}" ]] \
    || fail "binary was built from ${SHA} but expected ${EXPECTED_SHA} -- it is stale, rebuild it"
  pass "binary provenance matches ${EXPECTED_SHA}"
fi

# ------------------------------------------------------------------ /cfs2017
echo "==> GET /cfs2017"
MODELS="$(curl -fsS --max-time 5 "$BASE/cfs2017")"
MODEL_ID="$(printf '%s' "$MODELS" | "$PYTHON" -c '
import json,sys
data=json.load(sys.stdin).get("data",[])
if not data: sys.exit("no models exposed")
print(data[0]["main_id"])
')" || fail "no models exposed by /cfs2017"
pass "model exposed: $MODEL_ID"

# Derive the query from the model's own manifest so this can never drift from
# the served contract.
QUERY="$(printf '%s' "$MODELS" | "$PYTHON" -c '
import json,sys
from urllib.parse import urlencode
data=json.load(sys.stdin)["data"][0]
q={}
for col in data["input"]:
    if col["type"]=="numerical":
        q[col["name"]]=str((col["min"]+col["max"])//2)
    else:
        q[col["name"]]=str(col["available_values"][0])
print(urlencode(q))
')" || fail "could not derive a query from the model manifest"
pass "derived query: $QUERY"

# ---------------------------------------------------------------- inference
echo "==> GET /cfs2017/$MODEL_ID/inference"
BODY="$(curl -fsS --max-time 10 "$BASE/cfs2017/$MODEL_ID/inference?$QUERY")" \
  || fail "inference request failed"
echo "    $BODY"
printf '%s' "$BODY" | "$PYTHON" -c '
import json,math,sys
d=json.load(sys.stdin)
if d.get("message")!="success": sys.exit(f"unexpected message: {d}")
data=d.get("data") or {}
if not data: sys.exit("empty inference payload")
value=next(iter(data.values()))
value=float(value)
if not math.isfinite(value) or value <= 0:
    sys.exit(f"implausible prediction: {value}")
' || fail "inference returned an unusable value"
pass "inference returned a finite positive prediction"

# ------------------------------------------------------- error contract
echo "==> GET /cfs2017/__no_such_model__/inference (expect 404)"
CODE="$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 \
  "$BASE/cfs2017/__no_such_model__/inference?$QUERY")"
[[ "$CODE" == "404" ]] || fail "unknown model returned HTTP $CODE, expected 404"
pass "unknown model returns 404"

echo
echo "SMOKE TEST PASSED: $BINARY"
