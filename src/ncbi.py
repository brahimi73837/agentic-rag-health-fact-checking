import hashlib
import json
import logging
import time
from pathlib import Path
from typing import List, Dict, Optional
import requests

from src.config import NCBI_API_KEY, NCBI_EMAIL, NCBI_BASE, CACHE_DIR

log = logging.getLogger(__name__)
_LIVE_CACHE = CACHE_DIR / "pubmed_live"
_LIVE_CACHE.mkdir(parents=True, exist_ok=True)

_LAST_CALL = [0.0]
_MIN_INTERVAL = 0.12

def _throttle():
    elapsed = time.time() - _LAST_CALL[0]
    if elapsed < _MIN_INTERVAL:
        time.sleep(_MIN_INTERVAL - elapsed)
    _LAST_CALL[0] = time.time()

def _cache_key(*parts) -> Path:
    h = hashlib.sha1("||".join(str(p) for p in parts).encode()).hexdigest()[:16]
    return _LIVE_CACHE / f"{h}.json"

def esearch(query: str, retmax: int = 20) -> List[str]:
    ck = _cache_key("esearch", query, retmax)
    if ck.exists():
        return json.loads(ck.read_text())
    _throttle()
    params = {
        "db": "pubmed", "term": query, "retmax": retmax, "retmode": "json",
        "api_key": NCBI_API_KEY, "email": NCBI_EMAIL, "tool": "ThesisV3",
    }
    try:
        r = requests.get(f"{NCBI_BASE}/esearch.fcgi", params=params, timeout=15)
        r.raise_for_status()
        ids = r.json().get("esearchresult", {}).get("idlist", [])
        ck.write_text(json.dumps(ids))
        return ids
    except Exception as e:
        log.warning("esearch failed: %s", e)
        return []

def efetch_abstracts(pmids: List[str]) -> List[Dict[str, str]]:
    if not pmids:
        return []
    ck = _cache_key("efetch", ",".join(pmids))
    if ck.exists():
        return json.loads(ck.read_text())
    _throttle()
    params = {
        "db": "pubmed", "id": ",".join(pmids), "rettype": "abstract", "retmode": "xml",
        "api_key": NCBI_API_KEY, "email": NCBI_EMAIL, "tool": "ThesisV3",
    }
    try:
        r = requests.get(f"{NCBI_BASE}/efetch.fcgi", params=params, timeout=30)
        r.raise_for_status()
        docs = _parse_pubmed_xml(r.text)
        ck.write_text(json.dumps(docs))
        return docs
    except Exception as e:
        log.warning("efetch failed: %s", e)
        return []

def _parse_pubmed_xml(xml: str) -> List[Dict[str, str]]:
    import xml.etree.ElementTree as ET
    out = []
    try:
        root = ET.fromstring(xml)
    except ET.ParseError:
        return out
    for art in root.findall(".//PubmedArticle"):
        pmid_el = art.find(".//PMID")
        title_el = art.find(".//ArticleTitle")
        abs_nodes = art.findall(".//Abstract/AbstractText")
        pmid = pmid_el.text if pmid_el is not None else ""
        title = "".join(title_el.itertext()) if title_el is not None else ""
        abstract = " ".join("".join(n.itertext()) for n in abs_nodes) if abs_nodes else ""
        if pmid and (title or abstract):
            out.append({"pmid": pmid, "title": title.strip(), "abstract": abstract.strip()})
    return out

def search_and_fetch(query: str, retmax: int = 10) -> List[Dict[str, str]]:
    ids = esearch(query, retmax=retmax)
    if not ids:
        return []
    return efetch_abstracts(ids)
