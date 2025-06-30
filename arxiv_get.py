import os
import arxiv
import requests
import time

DOWNLOAD_DIR = "dataset/raw_data/PDF"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Referer": "https://arxiv.org"
}

def fetch_arxiv_papers(query="machine learning", max_results=10):
    search = arxiv.Search(query=query, max_results=max_results,
                          sort_by=arxiv.SortCriterion.SubmittedDate)
    for result in search.results():
        paper_id = result.get_short_id()  # e.g. 2406.12345v1
        pdf_url = f"https://arxiv.org/pdf/{paper_id}.pdf"
        save_path = os.path.join(DOWNLOAD_DIR, f"{paper_id}.pdf")

        if os.path.exists(save_path):
            print(f"⚠️ 已存在，跳过：{paper_id}")
            continue

        try:
            r = requests.get(pdf_url, headers=HEADERS, timeout=10)
            if r.status_code == 200:
                with open(save_path, "wb") as f:
                    f.write(r.content)
                print(f"✅ 下载成功: {paper_id}")
            else:
                print(f"❌ 下载失败 {paper_id}: 状态码 {r.status_code}")
        except Exception as e:
            print(f"❌ 异常 {paper_id}: {e}")
        time.sleep(2)

if __name__ == "__main__":
    fetch_arxiv_papers("deep learning", max_results=2)