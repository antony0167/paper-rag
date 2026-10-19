"""Getting papers from arXiv and turning PDFs into text."""

import re
import time
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List

import requests

ARXIV_API = "http://export.arxiv.org/api/query"
ATOM = {"a": "http://www.w3.org/2005/Atom"}
HEADERS = {"User-Agent": "paper-rag (github.com/antony0167/paper-rag)"}


@dataclass
class Paper:
    arxiv_id: str
    title: str
    authors: List[str]
    published: str
    abstract: str
    pdf_url: str

    def to_dict(self):
        return asdict(self)


def parse_atom(xml_text: str) -> List[Paper]:
    root = ET.fromstring(xml_text)
    papers = []
    for entry in root.findall("a:entry", ATOM):
        raw_id = entry.findtext("a:id", default="", namespaces=ATOM)
        arxiv_id = re.sub(r"v\d+$", "", raw_id.rsplit("/abs/", 1)[-1])
        pdf = next((l.get("href") for l in entry.findall("a:link", ATOM)
                    if l.get("title") == "pdf"), f"https://arxiv.org/pdf/{arxiv_id}")
        papers.append(Paper(
            arxiv_id=arxiv_id,
            title=" ".join(entry.findtext("a:title", "", ATOM).split()),
            authors=[a.findtext("a:name", "", ATOM) for a in entry.findall("a:author", ATOM)],
            published=entry.findtext("a:published", "", ATOM)[:10],
            abstract=" ".join(entry.findtext("a:summary", "", ATOM).split()),
            pdf_url=pdf,
        ))
    return papers


def fetch_metadata(ids: List[str] = None, query: str = None, max_results: int = 30) -> List[Paper]:
    """Either a list of arXiv ids or a search query."""
    if ids:
        params = {"id_list": ",".join(ids), "max_results": len(ids)}
    else:
        params = {"search_query": f"all:{query}", "max_results": max_results,
                  "sortBy": "relevance"}
    r = requests.get(ARXIV_API, params=params, headers=HEADERS, timeout=60)
    r.raise_for_status()
    return parse_atom(r.text)


def download_pdf(paper: Paper, folder: Path) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{paper.arxiv_id.replace('/', '_')}.pdf"
    if path.exists() and path.stat().st_size > 10_000:
        return path
    r = requests.get(paper.pdf_url, headers=HEADERS, timeout=120)
    r.raise_for_status()
    path.write_bytes(r.content)
    time.sleep(3)  # arXiv asks for a few seconds between requests
    return path


# ---------------------------------------------------------------- PDF -> text

_REFS = re.compile(r"^\s*(references|bibliography)\s*$", re.I | re.M)


def clean_page(text: str) -> str:
    text = text.replace("­", "")
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)      # re-join hyphenated line breaks
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def pdf_pages(path: Path) -> List[str]:
    """Text of each page, stopping at the reference list (it only adds noise)."""
    from pypdf import PdfReader

    pages = []
    for page in PdfReader(str(path)).pages:
        text = clean_page(page.extract_text() or "")
        m = _REFS.search(text)
        if m and len(pages) >= 3:
            pages.append(text[: m.start()])
            break
        pages.append(text)
    return pages
