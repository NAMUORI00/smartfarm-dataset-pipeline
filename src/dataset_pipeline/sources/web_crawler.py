"""Web crawler for wasabi cultivation documents.

Collects wasabi-related documents from various web sources:
- Academic PDFs (Oregon State, Czech University papers)
- Hydroponic wasabi guides (wasabicrop.co.uk)
- Korean agricultural papers (KCI, DBpia abstracts)
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional
from urllib.parse import urljoin, urlparse

import requests

# Optional PDF parsing
try:
    import pypdf
    HAS_PYPDF = True
except ImportError:
    HAS_PYPDF = False

# Optional HTML parsing
try:
    from bs4 import BeautifulSoup
    HAS_BS4 = True
except ImportError:
    HAS_BS4 = False


@dataclass(frozen=True)
class WebDoc:
    """A document fetched from the web."""
    id: str
    url: str
    title: str
    text: str
    metadata: Dict[str, Any]


# Curated list of wasabi cultivation sources
WASABI_SOURCES = {
    "academic_pdfs": [
        {
            "url": "https://agresearchfoundation.oregonstate.edu/sites/agresearchfoundation.oregonstate.edu/files/nackley_wasabi_interm_report_2022.pdf",
            "title": "Oregon State Wasabi Research Report 2022",
            "type": "pdf",
        },
        {
            "url": "https://acta.mendelu.cz/pdfs/acu/2019/01/33.pdf",
            "title": "Wasabi Propagation Techniques Review (Czech)",
            "type": "pdf",
        },
    ],
    "hydroponic_guides": [
        {
            "url": "https://wasabicrop.co.uk/growing-hydroponic-wasabi/",
            "title": "Growing Hydroponic Wasabi",
            "type": "html",
        },
        {
            "url": "https://wasabicrop.co.uk/growing-wasabi-in-hydroponic-systems/",
            "title": "Wasabi in Hydroponic Systems",
            "type": "html",
        },
        {
            "url": "https://wasabicrop.co.uk/from-seed-to-rhizome-mastering-the-art-of-hydroponic-wasabi-cultivation-techniques/",
            "title": "From Seed to Rhizome: Hydroponic Wasabi Cultivation",
            "type": "html",
        },
        {
            "url": "https://wasabicrop.co.uk/cultivating-the-coveted-green-treasure-an-expert-guide-to-growing-wasabi-indoors/",
            "title": "Growing Wasabi Indoors Guide",
            "type": "html",
        },
        {
            "url": "https://sproutandsow.com/growing-hydroponic-wasabi-a-complete-guide-for-success/",
            "title": "Complete Guide to Hydroponic Wasabi",
            "type": "html",
        },
        {
            "url": "https://pfboost.com/en/wasabi-plant/",
            "title": "Wasabi in Plant Factory Hydroponics",
            "type": "html",
        },
    ],
    "additional_guides": [
        {
            "url": "https://www.britannica.com/plant/wasabi",
            "title": "Wasabi - Britannica",
            "type": "html",
        },
        {
            "url": "https://en.wikipedia.org/wiki/Wasabi",
            "title": "Wasabi - Wikipedia",
            "type": "html",
        },
        {
            "url": "https://en.wikipedia.org/wiki/Eutrema_japonicum",
            "title": "Eutrema japonicum - Wikipedia",
            "type": "html",
        },
    ],
    "agriculture_extension": [
        {
            "url": "https://extension.oregonstate.edu/news/wasabi-viable-crop-willamette-valley",
            "title": "Wasabi as Viable Crop in Willamette Valley",
            "type": "html",
        },
    ],
}


def _clean_text(text: str) -> str:
    """Clean extracted text."""
    if not text:
        return ""
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove control characters
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
    return text.strip()


def _extract_text_from_pdf(content: bytes) -> str:
    """Extract text from PDF bytes."""
    if not HAS_PYPDF:
        return ""
    try:
        import io
        reader = pypdf.PdfReader(io.BytesIO(content))
        texts = []
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                texts.append(page_text)
        return "\n\n".join(texts)
    except Exception as e:
        print(f"PDF extraction error: {e}")
        return ""


def _extract_text_from_html(html: str, url: str) -> tuple[str, str]:
    """Extract title and main text from HTML."""
    if not HAS_BS4:
        # Fallback: simple regex extraction
        title_match = re.search(r'<title[^>]*>([^<]+)</title>', html, re.I)
        title = title_match.group(1) if title_match else ""
        # Remove script/style tags
        text = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.S | re.I)
        text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.S | re.I)
        text = re.sub(r'<[^>]+>', ' ', text)
        return title.strip(), _clean_text(text)
    
    soup = BeautifulSoup(html, 'html.parser')
    
    # Get title
    title = ""
    title_tag = soup.find('title')
    if title_tag:
        title = title_tag.get_text(strip=True)
    
    # Remove unwanted elements
    for tag in soup(['script', 'style', 'nav', 'header', 'footer', 'aside', 'iframe']):
        tag.decompose()
    
    # Try to find main content
    main_content = None
    for selector in ['article', 'main', '.content', '.post-content', '#content']:
        if selector.startswith('.') or selector.startswith('#'):
            main_content = soup.select_one(selector)
        else:
            main_content = soup.find(selector)
        if main_content:
            break
    
    if main_content:
        text = main_content.get_text(separator='\n', strip=True)
    else:
        # Fallback to body
        body = soup.find('body')
        text = body.get_text(separator='\n', strip=True) if body else soup.get_text(separator='\n', strip=True)
    
    return title, _clean_text(text)


def fetch_url(url: str, timeout: int = 30) -> Optional[bytes]:
    """Fetch URL content with proper headers."""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5,ko;q=0.3',
    }
    try:
        resp = requests.get(url, headers=headers, timeout=timeout, allow_redirects=True)
        resp.raise_for_status()
        return resp.content
    except Exception as e:
        print(f"Fetch error for {url}: {e}")
        return None


def crawl_wasabi_sources(
    categories: Optional[List[str]] = None,
    delay: float = 1.0,
) -> Iterator[WebDoc]:
    """Crawl curated wasabi sources.
    
    Args:
        categories: List of source categories to crawl. Default: all.
        delay: Delay between requests in seconds.
    
    Yields:
        WebDoc objects with extracted text.
    """
    if categories is None:
        categories = list(WASABI_SOURCES.keys())
    
    for category in categories:
        sources = WASABI_SOURCES.get(category, [])
        for source in sources:
            url = source["url"]
            doc_type = source.get("type", "html")
            default_title = source.get("title", "")
            
            print(f"Fetching: {url}")
            content = fetch_url(url)
            if not content:
                continue
            
            if doc_type == "pdf":
                text = _extract_text_from_pdf(content)
                title = default_title
            else:
                try:
                    html = content.decode('utf-8', errors='replace')
                except:
                    html = content.decode('latin-1', errors='replace')
                title, text = _extract_text_from_html(html, url)
                if not title:
                    title = default_title
            
            if not text or len(text) < 100:
                print(f"  Skipped (too short): {len(text) if text else 0} chars")
                continue
            
            # Generate doc ID from URL
            parsed = urlparse(url)
            doc_id = f"{parsed.netloc}{parsed.path}".replace('/', '_').replace('.', '_')[:100]
            
            yield WebDoc(
                id=doc_id,
                url=url,
                title=title,
                text=text,
                metadata={
                    "source": "web_crawler",
                    "category": category,
                    "lang": "en",
                    "url": url,
                },
            )
            
            print(f"  Extracted: {len(text)} chars")
            time.sleep(delay)


def iter_wasabi_web_docs(
    limit: Optional[int] = None,
    delay: float = 1.0,
) -> Iterator[WebDoc]:
    """Iterate over all wasabi web documents.
    
    Args:
        limit: Maximum number of documents to yield.
        delay: Delay between requests.
    
    Yields:
        WebDoc objects.
    """
    count = 0
    for doc in crawl_wasabi_sources(delay=delay):
        yield doc
        count += 1
        if limit and count >= limit:
            break


if __name__ == "__main__":
    # Quick test
    for doc in iter_wasabi_web_docs(limit=3):
        print(f"\n{'='*60}")
        print(f"ID: {doc.id}")
        print(f"Title: {doc.title}")
        print(f"URL: {doc.url}")
        print(f"Text preview: {doc.text[:500]}...")
