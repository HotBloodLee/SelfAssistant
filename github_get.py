import os, requests
from datetime import datetime, timedelta

import os, requests
from github import Github
from tqdm import tqdm

from core.utils.common_utils import load_secret, read_file

secret_config = load_secret("private/Secret.ini")

# ⚙️ 配置
GITHUB_TOKEN = secret_config["GITHUB"]["GITHUB_TOKEN"]  # 建议设置为环境变量
SEARCH_EXTENSIONS = ["md"]  # 搜索的文件扩展名列表
LANGS = {"md": "Markdown", "pdf": "PDF", "ppt": "PPT", "pptx": "PPT", "docx": "Word", "xlsx": "Excel"}
DOWNLOAD_DIR = "dataset/raw_data"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

# 初始化 GitHub API 客户端
gh = Github(GITHUB_TOKEN, per_page=30)

def search_and_download(ext, max_per_ext=1000):
    # 计算一年前的日期
    one_year_ago = datetime.now() - timedelta(days=365*5)
    one_year_ago_str = one_year_ago.strftime("%Y-%m-%d")

    query = f"extension:{ext}"  # 搜索指定扩展名的文件，且在一年内有更新
    print(f"\n搜索扩展名 .{ext} 的文件...")
    results = gh.search_code(query)  # 搜索代码内容包括文件路径

    count = 0
    for file in tqdm(results):
        if count >= max_per_ext: break
        try:
            raw = file.download_url
            name = os.path.basename(raw)
            path = os.path.join(DOWNLOAD_DIR, f"{file.repository.full_name.replace('/', '_')}_{name}")
            if os.path.exists(path):
                count += 1
                continue
            r = requests.get(raw, timeout=10)
            if r.status_code == 200:
                path = os.path.join(DOWNLOAD_DIR, f"{file.repository.full_name.replace('/','_')}_{name}")

                open(path, "wb").write(r.content)
                try:
                    check_file, _ = read_file(path)
                    if len(check_file) > 0:
                        # print(f"✅ 下载 {name} 来自 {file.repository.full_name}")
                        count += 1
                except Exception as e:
                    print(f"⚠️ 文件 {name} 可能无法解析: {e}")
                    # 删除文件
                    os.remove(path)
        except Exception as e:
            print(f"❌ 下载失败 {raw}: {e}")

# 遍历扩展名批量下载
for ext in SEARCH_EXTENSIONS:
    DOWNLOAD_DIR = f"dataset/raw_data/{LANGS[ext]}"
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)
    search_and_download(ext)
    break