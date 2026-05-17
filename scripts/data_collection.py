"""
Data Collection Script for Technology/IT Industry LLM Bot

This script collects:
1. Stack Overflow Q&A
2. GitHub Issues
3. Technical Documentation
4. LiveKit Documentation

Usage Examples:
    python scripts/data_collection.py --source livekit --limit 500
    python scripts/data_collection.py --source docs --limit 1000
    python scripts/data_collection.py --source all --limit 2500
"""

import argparse
import json
import time
from pathlib import Path
from typing import List, Dict, Any
from urllib.parse import urljoin

import pandas as pd
import requests
import trafilatura
from bs4 import BeautifulSoup
from tqdm import tqdm


class DataCollector:
    """Collect tech-specific data from multiple sources"""

    def __init__(self, output_dir: str = "data/raw"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.headers = {
            "User-Agent": "Mozilla/5.0"
        }

    # =========================================================
    # STACK OVERFLOW
    # =========================================================

    def collect_stackoverflow(self, limit: int = 1000) -> List[Dict[str, Any]]:
        """
        Collect Stack Overflow Q&A data
        """

        print(f"📚 Collecting Stack Overflow data (limit: {limit})...")

        base_url = "https://api.stackexchange.com/2.3/questions"

        tags = [
            "python",
            "javascript",
            "reactjs",
            "node.js",
            "docker",
            "kubernetes",
            "webrtc",
            "websocket",
            "fastapi",
            "livekit"
        ]

        data = []
        page_size = 50

        try:
            with tqdm(total=limit, desc="StackOverflow") as pbar:

                for tag in tags:

                    if len(data) >= limit:
                        break

                    params = {
                        "page": 1,
                        "pagesize": page_size,
                        "order": "desc",
                        "sort": "votes",
                        "tagged": tag,
                        "site": "stackoverflow",
                        "filter": "withbody"
                    }

                    response = requests.get(base_url, params=params)

                    if response.status_code != 200:
                        continue

                    questions = response.json().get("items", [])

                    for q in questions:

                        if len(data) >= limit:
                            break

                        if not q.get("accepted_answer_id"):
                            continue

                        answer_url = (
                            f"https://api.stackexchange.com/2.3/answers/"
                            f"{q['accepted_answer_id']}"
                        )

                        answer_response = requests.get(
                            answer_url,
                            params={
                                "site": "stackoverflow",
                                "filter": "withbody"
                            }
                        )

                        if answer_response.status_code != 200:
                            continue

                        answer_items = answer_response.json().get("items", [])

                        if not answer_items:
                            continue

                        answer_body = answer_items[0].get("body", "")

                        record = {
                            "source": "stackoverflow",
                            "question": self._clean_html(q.get("title", "")),
                            "context": self._clean_html(q.get("body", "")),
                            "answer": self._clean_html(answer_body),
                            "tags": q.get("tags", []),
                            "score": q.get("score", 0)
                        }

                        data.append(record)

                        pbar.update(1)

                        time.sleep(0.2)

                    time.sleep(1)

        except Exception as e:
            print(f"❌ StackOverflow error: {e}")

        print(f"✅ Collected {len(data)} StackOverflow samples")

        return data

    # =========================================================
    # GITHUB ISSUES
    # =========================================================

    def collect_github_issues(self, limit: int = 1000) -> List[Dict[str, Any]]:
        """
        Collect GitHub issues from popular repositories
        """

        print(f"🐙 Collecting GitHub issues (limit: {limit})...")

        repos = [
            "livekit/livekit",
            "livekit/agents",
            "microsoft/vscode",
            "facebook/react",
            "pallets/flask",
            "fastapi/fastapi",
            "docker/docker-ce"
        ]

        data = []

        try:
            with tqdm(total=limit, desc="GitHub") as pbar:

                for repo in repos:

                    if len(data) >= limit:
                        break

                    url = f"https://api.github.com/repos/{repo}/issues"

                    response = requests.get(
                        url,
                        headers=self.headers,
                        params={
                            "state": "closed",
                            "per_page": 50,
                            "sort": "comments"
                        }
                    )

                    if response.status_code != 200:
                        continue

                    issues = response.json()

                    for issue in issues:

                        if len(data) >= limit:
                            break

                        if "pull_request" in issue:
                            continue

                        record = {
                            "source": "github",
                            "question": issue.get("title", ""),
                            "context": issue.get("body", ""),
                            "answer": issue.get("body", ""),
                            "tags": [repo],
                            "score": issue.get("comments", 0),
                            "url": issue.get("html_url", "")
                        }

                        data.append(record)

                        pbar.update(1)

                    time.sleep(1)

        except Exception as e:
            print(f"❌ GitHub error: {e}")

        print(f"✅ Collected {len(data)} GitHub issues")

        return data

    # =========================================================
    # GENERAL TECH DOCS
    # =========================================================

    def collect_tech_docs(self, limit: int = 500) -> List[Dict[str, Any]]:
        """
        Collect general technical documentation
        """

        print(f"📖 Collecting technical docs (limit: {limit})...")

        urls = [
            "https://docs.python.org/3/tutorial/",
            "https://fastapi.tiangolo.com/",
            "https://flask.palletsprojects.com/en/stable/",
        ]

        data = []

        try:
            with tqdm(total=len(urls), desc="Tech Docs") as pbar:

                for url in urls:

                    if len(data) >= limit:
                        break

                    response = requests.get(url, headers=self.headers)

                    if response.status_code != 200:
                        continue

                    content = trafilatura.extract(response.text)

                    if not content:
                        continue

                    chunks = self._chunk_text(content)

                    for chunk in chunks:

                        if len(data) >= limit:
                            break

                        data.append({
                            "source": "documentation",
                            "question": "Technical Documentation",
                            "context": chunk,
                            "answer": chunk,
                            "tags": ["docs"],
                            "score": 1,
                            "url": url
                        })

                    pbar.update(1)

                    time.sleep(1)

        except Exception as e:
            print(f"❌ Docs error: {e}")

        print(f"✅ Collected {len(data)} doc samples")

        return data

    # =========================================================
    # LIVEKIT DOCS
    # =========================================================

    def collect_livekit_docs(self, limit: int = 1000) -> List[Dict[str, Any]]:
        """
        Collect LiveKit documentation data
        """

        print(f"🚀 Collecting LiveKit docs (limit: {limit})...")

        base_url = "https://docs.livekit.io"

        paths = [
            "/intro/overview/",
            "/agents/",
            "/agents/start/voice-ai/",
            "/agents/models/",
            "/agents/integrations/",
            "/home/client/connect/",
            "/home/client/tracks/",
            "/home/client/events/",
            "/reference/python/",
            "/reference/js/",
            "/reference/components/react/",
            "/recipes/",
            "/sip/",
        ]

        data = []

        try:
            with tqdm(total=len(paths), desc="LiveKit Docs") as pbar:

                for path in paths:

                    if len(data) >= limit:
                        break

                    full_url = urljoin(base_url, path)

                    try:
                        response = requests.get(
                            full_url,
                            headers=self.headers,
                            timeout=20
                        )

                        if response.status_code != 200:
                            continue

                        extracted = trafilatura.extract(
                            response.text,
                            include_links=False,
                            include_formatting=False
                        )

                        if not extracted:
                            continue

                        soup = BeautifulSoup(response.text, "html.parser")

                        title = (
                            soup.title.text.strip()
                            if soup.title
                            else "LiveKit Docs"
                        )

                        # Extract code snippets
                        code_blocks = []

                        for code in soup.find_all("code"):

                            code_text = code.get_text(strip=True)

                            if len(code_text) > 20:
                                code_blocks.append(code_text)

                        chunks = self._chunk_text(extracted)

                        for chunk in chunks:

                            if len(data) >= limit:
                                break

                            record = {
                                "source": "livekit_docs",
                                "question": title,
                                "context": chunk,
                                "answer": chunk,
                                "code_examples": code_blocks[:5],
                                "tags": [
                                    "livekit",
                                    "webrtc",
                                    "voice-ai",
                                    "realtime-ai"
                                ],
                                "url": full_url,
                                "score": 1
                            }

                            data.append(record)

                        pbar.update(1)

                        time.sleep(1)

                    except Exception as e:
                        print(f"❌ Error scraping {full_url}: {e}")

        except Exception as e:
            print(f"❌ LiveKit collection error: {e}")

        print(f"✅ Collected {len(data)} LiveKit samples")

        return data

    # =========================================================
    # HELPERS
    # =========================================================

    def _clean_html(self, text: str) -> str:
        """
        Remove HTML tags
        """

        if not text:
            return ""

        soup = BeautifulSoup(text, "html.parser")

        return soup.get_text(separator=" ", strip=True)

    def _chunk_text(self, text: str, chunk_size: int = 500) -> List[str]:
        """
        Split text into chunks
        """

        paragraphs = text.split("\n")

        cleaned = [
            p.strip()
            for p in paragraphs
            if len(p.strip()) > 50
        ]

        chunks = []

        current_chunk = ""

        for para in cleaned:

            if len(current_chunk) + len(para) < chunk_size:
                current_chunk += " " + para
            else:
                chunks.append(current_chunk.strip())
                current_chunk = para

        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks

    def save_data(self, data: List[Dict[str, Any]], filename: str):
        """
        Save data to JSON and CSV
        """

        output_path = self.output_dir / filename

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"💾 Saved JSON: {output_path}")

        df = pd.DataFrame(data)

        csv_path = output_path.with_suffix(".csv")

        df.to_csv(csv_path, index=False)

        print(f"💾 Saved CSV: {csv_path}")


# =========================================================
# MAIN
# =========================================================

def main():

    parser = argparse.ArgumentParser(
        description="Collect data for LLM training"
    )

    parser.add_argument(
        "--source",
        type=str,
        choices=[
            "stackoverflow",
            "github",
            "docs",
            "livekit",
            "all"
        ],
        default="all",
        help="Data source"
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=5000,
        help="Maximum records"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/raw",
        help="Output directory"
    )

    args = parser.parse_args()

    collector = DataCollector(output_dir=args.output_dir)

    all_data = []

    # =============================================
    # STACKOVERFLOW
    # =============================================

    if args.source in ["stackoverflow", "all"]:

        so_limit = (
            args.limit
            if args.source == "stackoverflow"
            else args.limit // 4
        )

        so_data = collector.collect_stackoverflow(limit=so_limit)

        all_data.extend(so_data)

        collector.save_data(
            so_data,
            "stackoverflow_data.json"
        )

    # =============================================
    # GITHUB
    # =============================================

    if args.source in ["github", "all"]:

        gh_limit = (
            args.limit
            if args.source == "github"
            else args.limit // 4
        )

        gh_data = collector.collect_github_issues(limit=gh_limit)

        all_data.extend(gh_data)

        collector.save_data(
            gh_data,
            "github_data.json"
        )

    # =============================================
    # DOCS
    # =============================================

    if args.source in ["docs", "all"]:

        doc_limit = (
            args.limit
            if args.source == "docs"
            else args.limit // 4
        )

        doc_data = collector.collect_tech_docs(limit=doc_limit)

        all_data.extend(doc_data)

        collector.save_data(
            doc_data,
            "docs_data.json"
        )

    # =============================================
    # LIVEKIT
    # =============================================

    if args.source in ["livekit", "all"]:

        lk_limit = (
            args.limit
            if args.source == "livekit"
            else args.limit // 4
        )

        lk_data = collector.collect_livekit_docs(limit=lk_limit)

        all_data.extend(lk_data)

        collector.save_data(
            lk_data,
            "livekit_docs.json"
        )

    # =============================================
    # SAVE COMBINED
    # =============================================

    if args.source == "all":

        collector.save_data(
            all_data,
            "combined_tech_data.json"
        )

    print("\n🎉 Data collection complete!")
    print(f"📦 Total records: {len(all_data)}")
    print(f"📁 Output directory: {args.output_dir}")


if __name__ == "__main__":
    main()