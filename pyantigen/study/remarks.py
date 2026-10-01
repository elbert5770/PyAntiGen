"""Remarks: the project's long-form reasoning, time-stamped, linked by id.

1.x kept its reasoning in code comments, some over a hundred lines long,
split across files and invisible where results are read. Here each piece of
reasoning is one entry in the project's ``remarks.json``:

    {"pyantigen_remarks": 1,
     "remarks": [
       {"id": "2026-09-23-why-drug-over-vehicle",
        "created": "2026-09-23T10:14:00",
        "title": "Why the drug arms are scored over vehicle",
        "body": "<p>...</p>",            # HTML
        "supersedes": null}
     ]}

A design object names the remarks that justify it (``Study(remarks=[...])``,
and in the design JSON as ``"remarks": ["2026-09-23-why-drug-over-vehicle"]``).
``validate`` refuses an id that is not in the file, and ``render_html``
writes ``remarks.html`` with one anchor per id, so the link
``remarks.html#<id>`` works from any report.

Entries are append-only. A changed opinion is a new remark whose
``supersedes`` names the old id; the old one stays, because results computed
under it still exist.
"""
import datetime as _dt
import html
import json
import os
import re

FORMAT = 1


def _slug(text):
    s = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return s[:48].rstrip("-") or "remark"


class Remarks:
    def __init__(self, path):
        self.path = path
        self.entries = []
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                doc = json.load(f)
            if doc.get("pyantigen_remarks") != FORMAT:
                raise ValueError(f"{path}: not a pyantigen remarks file (format {FORMAT})")
            self.entries = doc.get("remarks", [])
        ids = [e["id"] for e in self.entries]
        dup = {i for i in ids if ids.count(i) > 1}
        if dup:
            raise ValueError(f"{path}: duplicate remark ids {sorted(dup)}")

    @property
    def ids(self):
        return {e["id"] for e in self.entries}

    def get(self, rid):
        for e in self.entries:
            if e["id"] == rid:
                return e
        raise KeyError(f"no remark {rid!r} in {self.path}")

    def add(self, title, body_html, supersedes=None, now=None):
        """Append a remark and save. Returns its id (date + slug of title)."""
        now = now or _dt.datetime.now().replace(microsecond=0)
        base = f"{now:%Y-%m-%d}-{_slug(title)}"
        rid, n = base, 2
        while rid in self.ids:
            rid, n = f"{base}-{n}", n + 1
        if supersedes is not None and supersedes not in self.ids:
            raise KeyError(f"supersedes {supersedes!r}, which is not a remark")
        self.entries.append({"id": rid, "created": now.isoformat(),
                             "title": title, "body": body_html,
                             "supersedes": supersedes})
        self.save()
        return rid

    def save(self):
        with open(self.path, "w", encoding="utf-8", newline="\n") as f:
            json.dump({"pyantigen_remarks": FORMAT, "remarks": self.entries},
                      f, indent=1, ensure_ascii=False)
            f.write("\n")

    def link(self, rid, html_name="remarks.html"):
        self.get(rid)
        return f"{html_name}#{rid}"

    def render_html(self, out_path=None):
        out_path = out_path or os.path.join(os.path.dirname(self.path) or ".", "remarks.html")
        superseded_by = {e["supersedes"]: e["id"] for e in self.entries if e.get("supersedes")}
        parts = ["<!doctype html><meta charset='utf-8'><title>Remarks</title>",
                 "<style>body{font:15px/1.5 system-ui;max-width:48rem;margin:2rem auto;padding:0 1rem}"
                 "article{border-top:1px solid #ccc;padding:1rem 0}.meta{color:#666;font-size:13px}</style>",
                 "<h1>Remarks</h1>"]
        for e in sorted(self.entries, key=lambda e: e["created"], reverse=True):
            meta = f"{html.escape(e['created'])} &middot; <code>{html.escape(e['id'])}</code>"
            if e.get("supersedes"):
                meta += f" &middot; supersedes <a href='#{e['supersedes']}'>{e['supersedes']}</a>"
            if e["id"] in superseded_by:
                meta += f" &middot; superseded by <a href='#{superseded_by[e['id']]}'>{superseded_by[e['id']]}</a>"
            parts.append(f"<article id='{html.escape(e['id'])}'><h2>{html.escape(e['title'])}</h2>"
                         f"<p class='meta'>{meta}</p>{e['body']}</article>")
        with open(out_path, "w", encoding="utf-8", newline="\n") as f:
            f.write("\n".join(parts) + "\n")
        return out_path
