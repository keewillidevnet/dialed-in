#!/usr/bin/env python3
"""Fetch the C1 public-domain books from Project Gutenberg. Setup only.

Finds each book through the Gutendex catalog, downloads the plain text,
strips the Gutenberg header and footer, counts Glimmer tokens, and writes
a manifest with IDs and SHA-256 hashes. Nothing is run through the model
except the tokenizer.

    python3 fetch_books.py
"""
import hashlib
import json
import os
import sys
import time
import urllib.parse
import urllib.request

OUT = "/workspace/data/books"
TOKENIZER = "/workspace/weights/Muse-Glimmer-30B"
NEED_TOKENS = 33792  # 32,768 input tokens plus a 1,024-token margin

BOOKS = {
    "A": [("Pride and Prejudice", "Austen"), ("Moby Dick", "Melville"),
          ("Great Expectations", "Dickens"), ("Middlemarch", "Eliot"),
          ("The Count of Monte Cristo", "Dumas")],
    "B": [("Jane Eyre", "Bront"), ("Wuthering Heights", "Bront"),
          ("Dracula", "Stoker"), ("Frankenstein", "Shelley"),
          ("The Scarlet Letter", "Hawthorne")],
    "C": [("Little Women", "Alcott"), ("The Adventures of Tom Sawyer", "Twain"),
          ("Treasure Island", "Stevenson"), ("The Picture of Dorian Gray", "Wilde"),
          ("The Last of the Mohicans", "Cooper")],
}


def get(url, binary=False):
    req = urllib.request.Request(url, headers={"User-Agent": "dialed-in-setup/1.0"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                data = r.read()
            return data if binary else json.loads(data)
        except Exception as e:
            if attempt == 2:
                raise
            time.sleep(3)


def find_book(title, author):
    q = urllib.parse.urlencode({"search": f"{title} {author}", "languages": "en"})
    hits = get(f"https://gutendex.com/books/?{q}")["results"]
    want = title.lower().replace("-", " ")
    good = [b for b in hits
            if want in b["title"].lower().replace("-", " ")
            and any(author.lower() in a["name"].lower() for a in b["authors"])
            and any(k.startswith("text/plain") for k in b["formats"])]
    if not good:
        return None
    return max(good, key=lambda b: b.get("download_count", 0))


def strip_gutenberg(text):
    lines = text.splitlines()
    start = next((i for i, l in enumerate(lines) if l.startswith("*** START OF")), None)
    end = next((i for i, l in enumerate(lines) if l.startswith("*** END OF")), None)
    if start is None or end is None or end <= start:
        return None
    return "\n".join(lines[start + 1:end]).strip()


def main():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TOKENIZER)
    os.makedirs(OUT, exist_ok=True)
    manifest, problems = {"fetched_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                          "need_tokens": NEED_TOKENS, "books": []}, []

    for set_name, items in BOOKS.items():
        for title, author in items:
            book = find_book(title, author)
            if book is None:
                problems.append(f"{set_name}: not found: {title} ({author})")
                print(f"   MISSING {set_name} {title}")
                continue
            fmt = next(k for k in book["formats"] if k.startswith("text/plain") and "utf-8" in k) \
                if any("utf-8" in k for k in book["formats"] if k.startswith("text/plain")) \
                else next(k for k in book["formats"] if k.startswith("text/plain"))
            raw = get(book["formats"][fmt], binary=True)
            body = strip_gutenberg(raw.decode("utf-8", errors="replace"))
            if body is None:
                problems.append(f"{set_name}: no START/END markers: {book['title']}")
                continue
            n_tok = len(tok(body, add_special_tokens=False)["input_ids"])
            slug = f"{set_name}_{book['id']}"
            with open(os.path.join(OUT, f"{slug}.txt"), "w", encoding="utf-8") as fh:
                fh.write(body)
            entry = {"set": set_name, "gutenberg_id": book["id"], "title": book["title"],
                     "authors": [a["name"] for a in book["authors"]], "source_url": book["formats"][fmt],
                     "sha256_raw": hashlib.sha256(raw).hexdigest(),
                     "sha256_text": hashlib.sha256(body.encode("utf-8")).hexdigest(),
                     "tokens": n_tok, "long_enough": n_tok >= NEED_TOKENS, "file": f"{slug}.txt"}
            manifest["books"].append(entry)
            flag = "OK " if entry["long_enough"] else "SHORT"
            print(f"   {flag} {set_name}  #{book['id']:<6} {n_tok:>8,} tokens  {book['title'][:60]}")
            if not entry["long_enough"]:
                problems.append(f"{set_name}: too short ({n_tok} tokens): {book['title']}")

    manifest["problems"] = problems
    path = os.path.join(OUT, "MANIFEST.json")
    with open(path, "w") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"\n{'ALL 15 OK' if not problems and len(manifest['books']) == 15 else 'CHECK THESE:'}")
    for p in problems:
        print(f"   {p}")
    print(f"Manifest: {path}")
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
