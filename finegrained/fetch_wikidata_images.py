import argparse
import json
from pathlib import Path
import time

import requests
from requests.adapters import HTTPAdapter
import urllib
from urllib3.util.retry import Retry

WIKIDATA_API = "https://www.wikidata.org/w/api.php"
COMMONS_API = "https://commons.wikimedia.org/w/api.php"
_SESSION = requests.Session()
_SESSION.headers.update({
    "User-Agent": "FMUClip-WikidataFetcher/1.0 (contact: you@example.com)",
    "Accept": "application/json",
})

retry = Retry(
    total=10,
    connect=10,
    read=10,
    backoff_factor=1.2,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["GET"],
    respect_retry_after_header=True,
)
adapter = HTTPAdapter(max_retries=retry, pool_connections=50, pool_maxsize=50)
_SESSION.mount("https://", adapter)
_SESSION.mount("http://", adapter)

def http_get_json(url: str, timeout: int = 30, max_retries: int = 10, retry_backoff_sec: float = 1.5):
    # requests 的 timeout 可以拆成 (connect, read)，握手慢就加大 connect
    resp = _SESSION.get(url, timeout=(max(15, timeout), max(30, timeout)))
    resp.raise_for_status()
    return resp.json()


def wbsearchentities(query: str, limit: int = 5, timeout: int = 20, max_retries: int = 5):
    params = {
        "action": "wbsearchentities",
        "search": query,
        "language": "en",
        "format": "json",
        "limit": str(limit),
    }
    url = WIKIDATA_API + "?" + urllib.parse.urlencode(params)
    return http_get_json(url, timeout=timeout, max_retries=max_retries).get("search", [])


def wbgetentities(entity_id: str, timeout: int = 20, max_retries: int = 5):
    params = {
        "action": "wbgetentities",
        "ids": entity_id,
        "props": "claims|labels",
        "languages": "en",
        "format": "json",
    }
    url = WIKIDATA_API + "?" + urllib.parse.urlencode(params)
    return http_get_json(url, timeout=timeout, max_retries=max_retries).get("entities", {}).get(entity_id, {})


def extract_p18_file_titles(entity_obj):
    out = []
    claims = entity_obj.get("claims", {})
    for c in claims.get("P18", []):
        try:
            fname = c["mainsnak"]["datavalue"]["value"]
            if fname:
                out.append("File:" + fname if not fname.startswith("File:") else fname)
        except Exception:
            continue
    return out


def extract_p373_category(entity_obj):
    claims = entity_obj.get("claims", {})
    for c in claims.get("P373", []):
        try:
            return c["mainsnak"]["datavalue"]["value"]
        except Exception:
            continue
    return None


def commons_category_file_titles(category: str, limit: int = 200, timeout: int = 20, max_retries: int = 5):
    titles = []
    cmcontinue = None
    while len(titles) < limit:
        params = {
            "action": "query",
            "list": "categorymembers",
            "cmtitle": "Category:" + category,
            "cmnamespace": "6",
            "cmlimit": "50",
            "format": "json",
        }
        if cmcontinue:
            params["cmcontinue"] = cmcontinue
        url = COMMONS_API + "?" + urllib.parse.urlencode(params)
        data = http_get_json(url, timeout=timeout, max_retries=max_retries)
        for m in data.get("query", {}).get("categorymembers", []):
            t = m.get("title")
            if t and t.startswith("File:"):
                titles.append(t)
                if len(titles) >= limit:
                    break
        cmcontinue = data.get("continue", {}).get("cmcontinue")
        if not cmcontinue:
            break
    return titles


def commons_search_file_titles(query: str, limit: int = 200, timeout: int = 20, max_retries: int = 5):
    titles = []
    sroffset = 0
    while len(titles) < limit:
        params = {
            "action": "query",
            "list": "search",
            "srsearch": f'intitle:{query} filetype:bitmap',
            "srnamespace": "6",
            "srlimit": "50",
            "sroffset": str(sroffset),
            "format": "json",
        }
        url = COMMONS_API + "?" + urllib.parse.urlencode(params)
        data = http_get_json(url, timeout=timeout, max_retries=max_retries)
        batch = data.get("query", {}).get("search", [])
        if not batch:
            break
        for m in batch:
            t = m.get("title")
            if t and t.startswith("File:"):
                titles.append(t)
                if len(titles) >= limit:
                    break
        sroffset += len(batch)
        if len(batch) < 50:
            break
    return titles


def commons_file_url(file_title: str):
    # Redirects to the raw file URL.
    return "https://commons.wikimedia.org/wiki/Special:FilePath/" + urllib.parse.quote(
        file_title.replace("File:", ""), safe=""
    )


def download_file(
    url: str,
    out_path: Path,
    timeout: int = 30,
    max_retries: int = 5,
    retry_backoff_sec: float = 1.5,
):
    last_err = None
    for i in range(max_retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "FMUClip-WikidataFetcher/1.0"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = resp.read()
            out_path.write_bytes(data)
            return
        except (TimeoutError, urllib.error.URLError, urllib.error.HTTPError, OSError) as e:
            last_err = e
            if i < max_retries - 1:
                time.sleep(retry_backoff_sec * (2 ** i))
                continue
            raise
    if last_err is not None:
        raise last_err
    raise RuntimeError("Unexpected download_file failure")


def unique_keep_order(items):
    seen = set()
    out = []
    for x in items:
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out


def build_candidate_titles(
    concept: str,
    search_entities: int,
    need_num: int,
    sleep_sec: float,
    timeout_sec: int,
    max_retries: int,
):
    q = concept.replace("_", " ")
    entity_hits = wbsearchentities(q, limit=search_entities, timeout=timeout_sec, max_retries=max_retries)
    file_titles = []
    chosen_qids = []
    for hit in entity_hits:
        qid = hit.get("id")
        if not qid:
            continue
        chosen_qids.append(qid)
        ent = wbgetentities(qid, timeout=timeout_sec, max_retries=max_retries)
        file_titles.extend(extract_p18_file_titles(ent))
        cat = extract_p373_category(ent)
        if cat:
            file_titles.extend(
                commons_category_file_titles(
                    cat,
                    limit=300,
                    timeout=timeout_sec,
                    max_retries=max_retries,
                )
            )
        time.sleep(sleep_sec)

    if len(file_titles) < need_num:
        file_titles.extend(
            commons_search_file_titles(
                q,
                limit=500,
                timeout=timeout_sec,
                max_retries=max_retries,
            )
        )
    return unique_keep_order(file_titles), chosen_qids


def load_pred_rows_by_label(prediction_dir: Path):
    rows = {}
    for name in ["forget_test_topk.jsonl", "retain_test_topk.jsonl"]:
        p = prediction_dir / name
        if not p.exists():
            continue
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                label = r.get("label")
                if not label:
                    continue
                rows.setdefault(label, []).append(r)
    return rows


def main():
    parser = argparse.ArgumentParser("Fetch concept images from Wikidata/Commons.")
    parser.add_argument(
        "--concept_json_dir",
        type=str,
        default="/home/shenruoyan/FMUClip/data/classification/coco2017_instances/val/Df/item1",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="/home/shenruoyan/FMUClip/data/classification/coco2017_instances/val/test_images",
    )
    parser.add_argument(
        "--prediction_dir",
        type=str,
        default="/home/shenruoyan/FMUClip/finegrained/output/original_eval/original_banana3_100images_DF3_DR1_UNI3_03022152",
        help="Directory containing retain_test_topk.jsonl and forget_test_topk.jsonl.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="replace_wrong_only",
        choices=["replace_wrong_only", "full_fetch"],
        help="replace_wrong_only: only replace mispredicted samples in existing test_images; "
        "full_fetch: refetch full set for each concept.",
    )
    parser.add_argument("--num_per_concept", type=int, default=50)
    parser.add_argument("--search_entities", type=int, default=5)
    parser.add_argument("--sleep_sec", type=float, default=0.2)
    parser.add_argument("--clear_output_dir", action="store_true")
    parser.add_argument("--timeout_sec", type=int, default=30)
    parser.add_argument("--max_retries", type=int, default=6)
    parser.add_argument("--continue_on_concept_error", action="store_true")
    args = parser.parse_args()

    concept_dir = Path(args.concept_json_dir)
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    concept_files = sorted(concept_dir.glob("*.json"))
    summary = []
    pred_rows_by_label = {}
    if args.mode == "replace_wrong_only":
        pred_rows_by_label = load_pred_rows_by_label(Path(args.prediction_dir))

    for jf in concept_files:
        concept = jf.stem
        out_dir = out_root / concept
        out_dir.mkdir(parents=True, exist_ok=True)
        if args.clear_output_dir and args.mode == "full_fetch":
            for p in out_dir.glob("*"):
                if p.is_file():
                    p.unlink()

        if args.mode == "replace_wrong_only":
            rows = pred_rows_by_label.get(concept, [])
            wrong_indices = [i for i, r in enumerate(rows) if r.get("pred_name") != concept]
            files = sorted([p for p in out_dir.iterdir() if p.is_file()])
            if not rows:
                summary.append(
                    {
                        "concept": concept,
                        "saved": 0,
                        "target": 0,
                        "failed_downloads": 0,
                        "candidate_titles": 0,
                        "qids": [],
                        "note": "no_prediction_rows",
                    }
                )
                print(f"[{concept}] no prediction rows, skipped")
                continue
            if len(files) != len(rows):
                summary.append(
                    {
                        "concept": concept,
                        "saved": 0,
                        "target": len(wrong_indices),
                        "failed_downloads": 0,
                        "candidate_titles": 0,
                        "qids": [],
                        "note": f"file_count_mismatch files={len(files)} rows={len(rows)}",
                    }
                )
                print(f"[{concept}] file_count mismatch, skipped")
                continue
            if not wrong_indices:
                summary.append(
                    {
                        "concept": concept,
                        "saved": 0,
                        "target": 0,
                        "failed_downloads": 0,
                        "candidate_titles": 0,
                        "qids": [],
                        "note": "no_wrong_samples",
                    }
                )
                print(f"[{concept}] no wrong samples")
                continue

            try:
                file_titles, chosen_qids = build_candidate_titles(
                    concept,
                    args.search_entities,
                    len(wrong_indices),
                    args.sleep_sec,
                    args.timeout_sec,
                    args.max_retries,
                )
            except Exception as e:
                summary.append(
                    {
                        "concept": concept,
                        "saved": 0,
                        "target": len(wrong_indices),
                        "failed_downloads": 0,
                        "candidate_titles": 0,
                        "qids": [],
                        "note": f"candidate_fetch_error: {type(e).__name__}: {e}",
                    }
                )
                print(f"[{concept}] candidate fetch failed: {e}")
                if args.continue_on_concept_error:
                    continue
                raise
            saved = 0
            failed = 0
            for ft in file_titles:
                if saved >= len(wrong_indices):
                    break
                ext = Path(ft.replace("File:", "")).suffix.lower()
                if ext not in {".jpg", ".jpeg", ".png", ".webp"}:
                    continue
                slot = wrong_indices[saved]
                dst = files[slot]
                try:
                    download_file(
                        commons_file_url(ft),
                        dst,
                        timeout=args.timeout_sec,
                        max_retries=args.max_retries,
                    )
                    saved += 1
                except Exception:
                    failed += 1
                time.sleep(args.sleep_sec)

            summary.append(
                {
                    "concept": concept,
                    "saved": saved,
                    "target": len(wrong_indices),
                    "failed_downloads": failed,
                    "candidate_titles": len(file_titles),
                    "qids": chosen_qids,
                    "note": "replace_wrong_only",
                }
            )
            print(f"[{concept}] replaced={saved}/{len(wrong_indices)} candidates={len(file_titles)}")
            continue

        # full_fetch mode
        try:
            file_titles, chosen_qids = build_candidate_titles(
                concept,
                args.search_entities,
                args.num_per_concept,
                args.sleep_sec,
                args.timeout_sec,
                args.max_retries,
            )
        except Exception as e:
            summary.append(
                {
                    "concept": concept,
                    "saved": 0,
                    "target": args.num_per_concept,
                    "failed_downloads": 0,
                    "candidate_titles": 0,
                    "qids": [],
                    "note": f"candidate_fetch_error: {type(e).__name__}: {e}",
                }
            )
            print(f"[{concept}] candidate fetch failed: {e}")
            if args.continue_on_concept_error:
                continue
            raise

        saved = 0
        failed = 0
        for ft in file_titles:
            if saved >= args.num_per_concept:
                break
            ext = Path(ft.replace("File:", "")).suffix.lower()
            if ext not in {".jpg", ".jpeg", ".png", ".webp"}:
                continue
            dst = out_dir / f"{saved + 1:03d}_{Path(ft.replace('File:', '')).stem}{ext}"
            try:
                download_file(
                    commons_file_url(ft),
                    dst,
                    timeout=args.timeout_sec,
                    max_retries=args.max_retries,
                )
                saved += 1
            except Exception:
                failed += 1
            time.sleep(args.sleep_sec)

        summary.append(
            {
                "concept": concept,
                "saved": saved,
                "target": args.num_per_concept,
                "failed_downloads": failed,
                "candidate_titles": len(file_titles),
                "qids": chosen_qids,
            }
        )
        print(f"[{concept}] saved={saved}/{args.num_per_concept} candidates={len(file_titles)}")

    report = {
        "num_concepts": len(summary),
        "num_full": sum(1 for x in summary if x["saved"] >= x["target"]),
        "summary": summary,
    }
    rep_path = out_root / "_wikidata_fetch_report.json"
    rep_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"report: {rep_path}")


if __name__ == "__main__":
    main()
