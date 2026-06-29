"""Build a ponytail-style HTML comparison report from metrics.json + diffs/.

Reads ``metrics.json`` (next to this script) and the unified diffs in ``diffs/NN.diff``,
and writes ``report.html`` — a self-contained, dependency-free page (no CDN) with:
  - summary metric cards
  - horizontal bar charts (LOC, tokens, time, quality) with deltas vs the control run
  - a PASS/FAIL strip for the objective metadata test
  - a sortable comparison table
  - one collapsible, colour-coded git diff per run

Run:  python3 build_report.py
"""
import html
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def esc(s):
    return html.escape(str(s), quote=True)


def diff_to_html(path):
    """Render a unified diff as colour-coded HTML lines."""
    if not path or not os.path.isfile(path):
        return '<div class="nodiff">No code changes captured for this run.</div>'
    out = []
    with open(path, encoding="utf-8", errors="replace") as f:
        text = f.read()
    if not text.strip():
        return '<div class="nodiff">No code changes captured for this run.</div>'
    for line in text.splitlines():
        cls = "ctx"
        if line.startswith(("+++", "---")):
            cls = "meta"
        elif line.startswith("diff ") or line.startswith("index ") or line.startswith("@@"):
            cls = "hunk"
        elif line.startswith("+"):
            cls = "add"
        elif line.startswith("-"):
            cls = "del"
        out.append(f'<span class="dl {cls}">{esc(line)}</span>')
    return '<pre class="diff">' + "\n".join(out) + "</pre>"


def bar_chart(runs, key, label, unit="", lower_better=False, fmt="{:.0f}"):
    """Horizontal bar chart for one metric across runs, with deltas vs control."""
    vals = [(r.get(key) or 0) for r in runs]
    vmax = max(vals + [1])
    control = next((r for r in runs if r["name"] == "control"), None)
    cval = (control.get(key) or 0) if control else 0
    rows = []
    for r, v in zip(runs, vals):
        pct = 100.0 * v / vmax if vmax else 0
        delta = ""
        if control and r["name"] != "control" and cval:
            d = 100.0 * (v - cval) / cval
            sign = "+" if d >= 0 else ""
            good = (d < 0) if lower_better else (d > 0)
            delta = f'<span class="delta {"good" if good else "bad"}">{sign}{d:.0f}%</span>'
        is_ctrl = " ctrl" if r["name"] == "control" else ""
        rows.append(
            f'<div class="brow"><div class="blabel">{esc(r["label"])}</div>'
            f'<div class="btrack"><div class="bfill{is_ctrl}" style="width:{pct:.1f}%"></div></div>'
            f'<div class="bval">{esc(fmt.format(v))}{esc(unit)} {delta}</div></div>'
        )
    return (
        f'<div class="chart"><h3>{esc(label)}'
        f'{" (lower is better)" if lower_better else ""}</h3>{"".join(rows)}</div>'
    )


def main():
    meta = json.load(open(os.path.join(HERE, "metrics.json")))
    runs = meta["runs"]

    npass = sum(1 for r in runs if r.get("test_passed"))
    best_q = max(runs, key=lambda r: r.get("quality_overall") or 0)
    # cheapest passing run by tokens
    passing = [r for r in runs if r.get("test_passed")]
    cheapest = min(passing, key=lambda r: r.get("tokens") or 1e18) if passing else None

    cards = [
        ("Runs", f"{len(runs)}", "AGENTS.md variants (incl. control)"),
        ("Objective test", f"{npass}/{len(runs)} pass", "metadata embedded in complete config"),
        ("Top quality", f'{best_q["label"]}', f'rated {best_q.get("quality_overall","?")}/10'),
        (
            "Cheapest pass",
            (cheapest["label"] if cheapest else "—"),
            (f'{cheapest.get("tokens",0):,} tokens' if cheapest else "no run passed"),
        ),
    ]
    card_html = "".join(
        f'<div class="card"><div class="cval">{esc(v)}</div>'
        f'<div class="ckey">{esc(k)}</div><div class="csub">{esc(s)}</div></div>'
        for k, v, s in cards
    )

    charts = (
        bar_chart(runs, "quality_overall", "Code-quality rating", unit="/10", fmt="{:.1f}")
        + bar_chart(runs, "loc_total", "Lines changed (added + removed)", lower_better=False)
        + bar_chart(runs, "tokens", "Tokens used", lower_better=True, fmt="{:,.0f}")
        + bar_chart(runs, "time_s", "Wall-clock time", unit="s", lower_better=True)
    )

    # PASS/FAIL strip
    strip = "".join(
        f'<div class="chip {"pass" if r.get("test_passed") else "fail"}" '
        f'title="coverage {r.get("coverage",0):.0%}">{esc(r["label"])}<br>'
        f'{"PASS" if r.get("test_passed") else "FAIL"} · {r.get("coverage",0):.0%}</div>'
        for r in runs
    )

    # Sortable table
    cols = [
        ("label", "Variant", "t"),
        ("variant_lines", "Instr. lines", "n"),
        ("test_passed", "Test", "b"),
        ("coverage", "Coverage", "p"),
        ("quality_overall", "Quality", "n"),
        ("loc_total", "LOC", "n"),
        ("files_changed", "Files", "n"),
        ("tokens", "Tokens", "n"),
        ("tool_uses", "Tools", "n"),
        ("time_s", "Time(s)", "n"),
    ]
    thead = "".join(f'<th data-k="{k}" data-t="{t}">{esc(h)} ⇅</th>' for k, h, t in cols)
    trows = []
    for r in runs:
        tds = []
        for k, h, t in cols:
            v = r.get(k)
            if k == "test_passed":
                cell = f'<span class="tag {"pass" if v else "fail"}">{"PASS" if v else "FAIL"}</span>'
            elif k == "coverage":
                cell = f"{(v or 0):.0%}"
            elif k == "tokens":
                cell = f"{(v or 0):,}"
            else:
                cell = esc(v if v is not None else "—")
            tds.append(f"<td>{cell}</td>")
        trows.append(f'<tr class="{"ctrlrow" if r["name"]=="control" else ""}">' + "".join(tds) + "</tr>")

    # Per-iteration detail sections
    details = []
    for r in runs:
        q = (
            f'correctness {r.get("quality_correctness","?")}, '
            f'completeness {r.get("quality_completeness","?")}, '
            f'style {r.get("quality_style","?")}, '
            f'adherence {r.get("quality_adherence","?")}'
        )
        details.append(
            f'<section class="iter"><h3>{esc(r["label"])} '
            f'<span class="tag {"pass" if r.get("test_passed") else "fail"}">'
            f'{"PASS" if r.get("test_passed") else "FAIL"}</span></h3>'
            f'<div class="metarow">'
            f'<span>instr lines: <b>{esc(r.get("variant_lines","—"))}</b></span>'
            f'<span>quality: <b>{esc(r.get("quality_overall","—"))}/10</b></span>'
            f'<span>coverage: <b>{r.get("coverage",0):.0%}</b></span>'
            f'<span>LOC: <b>{esc(r.get("loc_total","—"))}</b> ({esc(r.get("files_changed","—"))} files)</span>'
            f'<span>tokens: <b>{(r.get("tokens") or 0):,}</b></span>'
            f'<span>tools: <b>{esc(r.get("tool_uses","—"))}</b></span>'
            f'<span>time: <b>{esc(r.get("time_s","—"))}s</b></span></div>'
            f'<p class="qjust"><b>Quality ({q}):</b> {esc(r.get("quality_justification",""))}</p>'
            f'<p class="summary"><b>Agent summary:</b> {esc(r.get("summary",""))}</p>'
            f'<details><summary>Show code diff (old → new)</summary>{diff_to_html(os.path.join(HERE, r.get("diff_file","")) if r.get("diff_file") else None)}</details>'
            f"</section>"
        )

    page = TEMPLATE.format(
        issue=esc(meta.get("issue", "")),
        model=esc(meta.get("model", "")),
        generated=esc(meta.get("generated", "")),
        cards=card_html,
        charts=charts,
        strip=strip,
        thead=thead,
        trows="".join(trows),
        details="".join(details),
        caveat=esc(meta.get("caveat", "")),
    )
    out = os.path.join(HERE, "report.html")
    with open(out, "w", encoding="utf-8") as f:
        f.write(page)
    print("wrote", out)


TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AGENTS.md vs dingo #327 — comparison</title>
<style>
:root{{--bg:#0e1116;--panel:#171c24;--panel2:#1e252f;--ink:#e6edf3;--mut:#9aa7b4;--line:#2a323d;--accent:#6ea8fe;--add:#1f7a3d;--del:#9b2c2c;--good:#3fb950;--bad:#f85149;}}
@media (prefers-color-scheme:light){{:root{{--bg:#f6f8fa;--panel:#fff;--panel2:#f0f3f6;--ink:#1f2328;--mut:#5b6570;--line:#d0d7de;--accent:#0969da;--good:#1a7f37;--bad:#cf222e;}}}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}}
.wrap{{max-width:1040px;margin:0 auto;padding:32px 20px 80px}}
header h1{{margin:0 0 4px;font-size:26px}}
.sub{{color:var(--mut);margin:0 0 24px}}
.sub code{{background:var(--panel2);padding:1px 6px;border-radius:5px}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:14px;margin:22px 0}}
.card{{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:16px}}
.cval{{font-size:22px;font-weight:700}}
.ckey{{color:var(--ink);font-weight:600;margin-top:6px}}
.csub{{color:var(--mut);font-size:13px}}
h2{{margin:34px 0 12px;font-size:18px;border-bottom:1px solid var(--line);padding-bottom:6px}}
.chart{{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:14px 16px;margin:14px 0}}
.chart h3{{margin:0 0 12px;font-size:14px;color:var(--mut);font-weight:600}}
.brow{{display:grid;grid-template-columns:165px 1fr 150px;align-items:center;gap:10px;margin:5px 0}}
.blabel{{font-size:12.5px;color:var(--mut);text-align:right;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}}
.btrack{{background:var(--panel2);border-radius:6px;height:16px;overflow:hidden}}
.bfill{{height:100%;background:linear-gradient(90deg,var(--accent),#9b8cff);border-radius:6px}}
.bfill.ctrl{{background:linear-gradient(90deg,#6b7681,#8a96a3)}}
.bval{{font-size:12.5px;font-variant-numeric:tabular-nums}}
.delta{{font-size:11px;margin-left:4px}}
.delta.good{{color:var(--good)}} .delta.bad{{color:var(--bad)}}
.strip{{display:flex;flex-wrap:wrap;gap:8px;margin:12px 0}}
.chip{{flex:1 1 120px;text-align:center;padding:8px;border-radius:9px;font-size:12px;font-weight:600;border:1px solid var(--line)}}
.chip.pass{{background:rgba(63,185,80,.14);color:var(--good)}}
.chip.fail{{background:rgba(248,81,73,.14);color:var(--bad)}}
table{{width:100%;border-collapse:collapse;background:var(--panel);border:1px solid var(--line);border-radius:12px;overflow:hidden;font-size:13px}}
th,td{{padding:8px 10px;text-align:right;border-bottom:1px solid var(--line);white-space:nowrap}}
th:first-child,td:first-child{{text-align:left}}
th{{cursor:pointer;color:var(--mut);font-weight:600;user-select:none;background:var(--panel2)}}
tr.ctrlrow{{opacity:.75;font-style:italic}}
.tag{{padding:1px 7px;border-radius:20px;font-size:11px;font-weight:700}}
.tag.pass{{background:rgba(63,185,80,.16);color:var(--good)}}
.tag.fail{{background:rgba(248,81,73,.16);color:var(--bad)}}
.iter{{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:16px 18px;margin:14px 0}}
.iter h3{{margin:0 0 10px;font-size:16px}}
.metarow{{display:flex;flex-wrap:wrap;gap:14px;color:var(--mut);font-size:12.5px;margin-bottom:8px}}
.metarow b{{color:var(--ink)}}
.qjust,.summary{{font-size:13.5px;margin:6px 0}}
details{{margin-top:10px}} summary{{cursor:pointer;color:var(--accent);font-size:13px}}
pre.diff{{background:var(--bg);border:1px solid var(--line);border-radius:8px;padding:10px;overflow:auto;font:12px/1.45 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;max-height:480px;margin-top:8px}}
.dl{{display:block;white-space:pre}}
.dl.add{{background:rgba(63,185,80,.13);color:var(--good)}}
.dl.del{{background:rgba(248,81,73,.13);color:var(--bad)}}
.dl.hunk{{color:var(--accent)}} .dl.meta{{color:var(--mut)}}
.nodiff{{color:var(--mut);font-size:13px;padding:8px}}
.note{{color:var(--mut);font-size:13px;background:var(--panel);border:1px solid var(--line);border-left:3px solid var(--accent);border-radius:8px;padding:12px 14px;margin:14px 0}}
footer{{color:var(--mut);font-size:12px;margin-top:40px;text-align:center}}
</style></head>
<body><div class="wrap">
<header>
<h1>How AGENTS.md content shapes a fix</h1>
<p class="sub">Same task — <b>{issue}</b> — solved 11 times, each with a different AGENTS.md / agent config. Model: <code>{model}</code>. Generated {generated}.</p>
</header>
<div class="cards">{cards}</div>
<h2>Objective test — metadata embedded in the complete config</h2>
<div class="strip">{strip}</div>
<h2>Metrics</h2>
{charts}
<h2>Comparison table</h2>
<table id="tbl"><thead><tr>{thead}</tr></thead><tbody>{trows}</tbody></table>
<div class="note">{caveat}</div>
<h2>Per-iteration detail (with diffs)</h2>
{details}
<footer>Each run worked in an isolated git worktree off the same commit. Tokens/tool-uses/time are per fix-agent as reported by the harness; LOC is from the worktree diff (excluding the instruction-file swap); the objective test and quality rating are run separately.</footer>
</div>
<script>
const tbl=document.getElementById('tbl');
tbl.querySelectorAll('th').forEach((th,i)=>{{th.addEventListener('click',()=>{{
 const t=th.dataset.t, rows=[...tbl.tBodies[0].rows];
 const asc=th._asc=!th._asc;
 rows.sort((a,b)=>{{let x=a.cells[i].innerText,y=b.cells[i].innerText;
  if(t==='n'||t==='p'){{x=parseFloat(x.replace(/[^0-9.\-]/g,''))||0;y=parseFloat(y.replace(/[^0-9.\-]/g,''))||0;return asc?x-y:y-x;}}
  if(t==='b'){{x=x.includes('PASS')?1:0;y=y.includes('PASS')?1:0;return asc?x-y:y-x;}}
  return asc?x.localeCompare(y):y.localeCompare(x);}});
 rows.forEach(r=>tbl.tBodies[0].appendChild(r));
}});}});
</script>
</body></html>
"""


if __name__ == "__main__":
    main()
