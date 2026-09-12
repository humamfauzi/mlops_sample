#!/usr/bin/env python3
"""Registry maintenance for the SQLite model store.

`example.db` is the whole system: every run, metric, model and preprocessing
pipeline. It is also a single mutable file, which makes it awkward to version
with DVC -- DVC addresses whole files by content hash, so appending a few
kilobytes of metrics would re-upload the entire database. These commands
support the alternative: snapshot it consistently, and keep it small.

Commands:
    report    <db>                  size and composition
    verify    <db>                  integrity plus the serving invariant
    snapshot  <db> <dest>           consistent copy (VACUUM INTO)
    prune     <src> <dest>          copy, optionally dropping unloadable models

`snapshot` and `prune` both use SQLite's `VACUUM INTO`, which produces a
transactionally consistent copy **while the server is running**. Copying the
file with `cp` can capture a torn write and silently corrupt the result.

`prune` never modifies its source. It writes a new database and reports what it
dropped, so the original stays available until you are satisfied.
"""
from __future__ import annotations

import argparse
import os
import pathlib
import sqlite3
import sys

MIB = 1024 * 1024
MODEL_PREFIX = "model/"
MODEL_PREFIX_LEN = len(MODEL_PREFIX)

# A model blob is loadable only through its child run's `level=best` tag; the
# server resolves published runs with find_tagged_best_model, and post_test
# replays the same path. Any other child's pickle is never read by anything.
_UNLOADABLE_MODELS = f"""
    SELECT b.id, b.intent, LENGTH(b.data) AS bytes
    FROM blobs b
    WHERE b.intent LIKE '{MODEL_PREFIX}%'
      AND NOT EXISTS (
          SELECT 1 FROM runs rc
          JOIN tags t ON t.run_id = rc.id AND t.key = 'level' AND t.value = 'best'
          WHERE rc.name = SUBSTR(b.intent, {MODEL_PREFIX_LEN + 1})
      )
"""

# ...but only when the run has a better sibling. If a parent produced no best
# child at all -- a run that failed part way -- its models are the only record
# of what it managed to train, so keep them.
_HAS_BEST_SIBLING = f"""
      AND EXISTS (
          SELECT 1 FROM runs rc2
          WHERE rc2.name = SUBSTR(b.intent, {MODEL_PREFIX_LEN + 1})
            AND EXISTS (
                SELECT 1 FROM runs sib
                JOIN tags st ON st.run_id = sib.id
                             AND st.key = 'level' AND st.value = 'best'
                WHERE sib.parent_id = rc2.parent_id
            )
      )
"""

_PUBLISHED_INVARIANT = """
    SELECT rp.id, rp.name,
           (SELECT COUNT(*) FROM runs rc JOIN tags t ON t.run_id = rc.id
             WHERE rc.parent_id = rp.id AND t.key = 'level' AND t.value = 'best')
               AS best_children,
           (SELECT COUNT(*) FROM runs rc2 JOIN tags t2 ON t2.run_id = rc2.id
              JOIN blobs b ON b.intent = 'model/' || rc2.name
             WHERE rc2.parent_id = rp.id AND t2.key = 'level' AND t2.value = 'best')
               AS loadable
    FROM runs rp
    JOIN tags tp ON tp.run_id = rp.id
    WHERE tp.key = 'status.deployment' AND tp.value = 'published'
"""


def connect(path: str, readonly: bool = False) -> sqlite3.Connection:
    if readonly:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    else:
        conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def _vacuum_into(src: str, dest: str, force: bool) -> None:
    target = pathlib.Path(dest)
    if target.exists():
        if not force:
            raise SystemExit(
                f"refusing to overwrite {dest} (pass --force, or choose another name)"
            )
        target.unlink()
    target.parent.mkdir(parents=True, exist_ok=True)
    # Read-only connection: VACUUM INTO only reads the source.
    with connect(src, readonly=True) as conn:
        conn.execute("VACUUM INTO ?", (str(target),))


# --------------------------------------------------------------------- report
def cmd_report(args) -> int:
    path = args.db
    if not os.path.exists(path):
        raise SystemExit(f"no such database: {path}")
    with connect(path, readonly=True) as conn:
        total = os.path.getsize(path)
        rows = conn.execute(
            """SELECT
                 CASE
                   WHEN intent LIKE 'model/%' THEN 'model'
                   WHEN intent LIKE 'transformation_object/%' THEN 'transformation object'
                   WHEN intent = 'transformation_instruction' THEN 'transformation instruction'
                   ELSE 'other'
                 END AS kind,
                 COUNT(*) AS n, COALESCE(SUM(LENGTH(data)), 0) AS bytes
               FROM blobs GROUP BY kind ORDER BY bytes DESC"""
        ).fetchall()
        meta = conn.execute(
            """SELECT
                 (SELECT COALESCE(SUM(LENGTH(CAST(value AS BLOB))),0) FROM metrics) +
                 (SELECT COALESCE(SUM(LENGTH(CAST(value AS BLOB))),0) FROM properties) +
                 (SELECT COALESCE(SUM(LENGTH(CAST(value AS BLOB))),0) FROM tags) +
                 (SELECT COALESCE(SUM(LENGTH(url)),0) FROM objects)"""
        ).fetchone()[0]
        biggest = conn.execute(
            "SELECT intent, LENGTH(data) AS n FROM blobs ORDER BY n DESC LIMIT 5"
        ).fetchall()
        counts = {
            t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
            for t in ("runs", "metrics", "tags", "properties", "objects", "blobs", "experiments")
        }

    print(f"{path}  ({total / MIB:.1f} MiB)")
    print()
    print(f"  {'blobs':<26}{'count':>6}{'size':>12}")
    for r in rows:
        print(f"  {r['kind']:<26}{r['n']:>6}{r['bytes'] / MIB:>10.1f} MiB")
    blob_total = sum(r["bytes"] for r in rows) or 1
    print(f"  {'all metadata combined':<26}{'':>6}{meta / MIB:>10.4f} MiB"
          f"   ({100 * meta / total:.3f}% of the file)")
    print()
    print("  largest blobs:")
    for r in biggest:
        print(f"    {r['n'] / MIB:>9.1f} MiB  {r['intent']}")
    print()
    print("  rows: " + "  ".join(f"{k}={v}" for k, v in counts.items()))
    return 0


# --------------------------------------------------------------------- verify
def cmd_verify(args) -> int:
    with connect(args.db, readonly=True) as conn:
        integrity = conn.execute("PRAGMA integrity_check").fetchone()[0]
        print(f"  integrity_check: {integrity}")
        failures = 0
        published = conn.execute(_PUBLISHED_INVARIANT).fetchall()
        if not published:
            print("  published models: none")
        for r in published:
            ok = r["best_children"] == 1 and r["loadable"] == 1
            failures += 0 if ok else 1
            print(
                f"  published {r['name']}: best_children={r['best_children']} "
                f"loadable={r['loadable']} {'OK' if ok else 'BROKEN'}"
            )
    if integrity != "ok" or failures:
        print("\nVERIFY FAILED")
        return 1
    print("\nVERIFY OK")
    return 0


# ------------------------------------------------------------------- snapshot
def cmd_snapshot(args) -> int:
    _vacuum_into(args.db, args.dest, args.force)
    src, dst = os.path.getsize(args.db), os.path.getsize(args.dest)
    print(f"  snapshot written: {args.dest}  ({dst / MIB:.1f} MiB, source {src / MIB:.1f} MiB)")
    print("  taken with VACUUM INTO -- consistent even if the server is running")
    return 0


# ---------------------------------------------------------------------- prune
def cmd_prune(args) -> int:
    query = _UNLOADABLE_MODELS + ("" if args.include_incomplete else _HAS_BEST_SIBLING)
    with connect(args.src, readonly=True) as conn:
        victims = conn.execute(query).fetchall()
        total = conn.execute("SELECT COALESCE(SUM(LENGTH(data)),0) FROM blobs").fetchone()[0]

    freed = sum(v["bytes"] for v in victims)
    print(f"  {len(victims)} model blob(s) are not loadable by any code path")
    for v in sorted(victims, key=lambda v: -v["bytes"])[:10]:
        print(f"    {v['bytes'] / MIB:>9.1f} MiB  {v['intent']}")
    if len(victims) > 10:
        print(f"    ... and {len(victims) - 10} more")
    print()
    print(f"  blobs now {total / MIB:.1f} MiB -> about {(total - freed) / MIB:.1f} MiB "
          f"(frees {freed / MIB:.1f} MiB)")

    if args.dry_run:
        print("\n  --dry-run: nothing written")
        return 0

    print(f"\n  building pruned copy at {args.dest} ...")
    _vacuum_into(args.src, args.dest, args.force)
    with connect(args.dest) as conn:
        conn.executemany("DELETE FROM blobs WHERE id = ?", [(v["id"],) for v in victims])
        deleted = conn.total_changes
        conn.commit()
        conn.execute("VACUUM")
        conn.commit()
        failures = [
            r for r in conn.execute(_PUBLISHED_INVARIANT).fetchall()
            if r["best_children"] != 1 or r["loadable"] != 1
        ]

    size = os.path.getsize(args.dest)
    print(f"  deleted {deleted} blob(s); result is {size / MIB:.1f} MiB")
    if failures:
        print("\n  PRUNE WOULD BREAK SERVING:")
        for r in failures:
            print(f"    published {r['name']} has no loadable model")
        print("  leaving the source untouched; delete the output and investigate")
        return 1
    print("\n  published models still loadable. Source is untouched.")
    print(f"  to adopt: verify {args.dest}, then move it over {args.src}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Registry maintenance for the SQLite model store.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("report", help="size and composition")
    p.add_argument("db")
    p.set_defaults(func=cmd_report)

    p = sub.add_parser("verify", help="integrity plus the serving invariant")
    p.add_argument("db")
    p.set_defaults(func=cmd_verify)

    p = sub.add_parser("snapshot", help="consistent copy via VACUUM INTO")
    p.add_argument("db")
    p.add_argument("dest")
    p.add_argument("--force", action="store_true", help="overwrite an existing destination")
    p.set_defaults(func=cmd_snapshot)

    p = sub.add_parser("prune", help="copy without models nothing can load")
    p.add_argument("src")
    p.add_argument("dest")
    p.add_argument("--dry-run", action="store_true", help="report only; write nothing")
    p.add_argument("--force", action="store_true", help="overwrite an existing destination")
    p.add_argument(
        "--include-incomplete",
        action="store_true",
        help="also drop models from runs that produced no best child at all",
    )
    p.set_defaults(func=cmd_prune)

    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
