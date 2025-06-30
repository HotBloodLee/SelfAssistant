# coding: utf-8
import os
import requests
from github import Github
from urllib.parse import urljoin

from tqdm import tqdm

from core.utils.common_utils import load_secret, load_json

# 配置
secret_config = load_secret("private/Secret.ini")
repo_config = load_json("config/repo.json")

# ⚙️ 配置
GITHUB_TOKEN = secret_config["GITHUB"]["GITHUB_TOKEN"] # 建议设置为环境变量
REPO_NAMES = repo_config["PPT"]
TARGET_EXTENSIONS = [".pdf", ".pptx", ".docx", ".xlsx", ".md", ".ppt", ".doc"]  # 需要下载的文件扩展名列表
LANGS = {".md": "Markdown", ".pdf": "PDF", ".ppt": "PPT", ".pptx": "PPT", ".docx": "Word", ".doc": "Word", ".xlsx": "Excel"}
DOWNLOAD_DIR = "dataset/raw_data"

os.makedirs(DOWNLOAD_DIR, exist_ok=True)

# 初始化 GitHub API
gh = Github(GITHUB_TOKEN, per_page=100)

def download_file(url, dest_path):
    try:
        r = requests.get(url, stream=True, timeout=10)
        r.raise_for_status()
        with open(dest_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"✅ 下载成功: {dest_path}")
    except Exception as e:
        print(f"❌ 下载失败 {url}: {e}")

def process_repo(repo_full_name):
    print(f"\n🔍 处理仓库: {repo_full_name}")
    try:
        repo = gh.get_repo(repo_full_name)
        contents = repo.get_contents("")
        stack = contents[:]

        while stack:
            file = stack.pop()
            if file.type == "dir":
                stack.extend(repo.get_contents(file.path))
            elif any(file.name.lower().endswith(ext) for ext in TARGET_EXTENSIONS):
                raw_url = file.download_url
                name = os.path.basename(raw_url)
                type_ = "Other"
                for ext in TARGET_EXTENSIONS:
                    if file.name.lower().endswith(ext):
                        type_ = LANGS[ext]
                os.makedirs(DOWNLOAD_DIR + "/" + type_, exist_ok=True)
                filename = repo_full_name.replace("/", "_") + "_" + os.path.basename(file.path)

                dest_path = os.path.join(DOWNLOAD_DIR + "/" + type_, filename)
                if os.path.exists(dest_path):
                    continue
                download_file(raw_url, dest_path)

    except Exception as e:
        print(f"❌ 仓库处理失败: {repo_full_name}, 错误: {e}")

# 主程序
if __name__ == "__main__":
    for repo_name in tqdm(REPO_NAMES):
        process_repo(repo_name)
    print("\n✅ 所有仓库处理完成。文件保存于:", DOWNLOAD_DIR)