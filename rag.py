import requests
from bs4 import BeautifulSoup
import time
import pandas as pd
import os
import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# openai公式ライブラリ
import openai

# ---------------------------------------
# 1) 初期設定
# ---------------------------------------
OPENAI_API_KEY= 'sk-xxxx...'  # ご自身のAPIキーを設定
openai.api_key = OPENAI_API_KEY

# Embeddingに使用するモデル
# 公開されているモデル例: "text-embedding-ada-002"
EMBEDDING_MODEL = 'text-embedding-3-small'  # 必要に応じて "text-embedding-ada-002" に変更

# Chatに使用するモデル
GPT_MODEL = "gpt-3.5-turbo"

# サイトマップのURL
SITEMAP_URL = 'https://jinjabukkaku.online/post-sitemap.xml'

# 出力ファイル
CSV_FILE = 'summarized_content.csv'
BIN_FILE = 'embeddings.bin'

# ---------------------------------------
# 2) サイトマップのURL一覧取得
# ---------------------------------------
def fetch_sitemap_urls(sitemap_url):
    response = requests.get(sitemap_url)
    soup = BeautifulSoup(response.content, 'xml')
    urls = [loc.text for loc in soup.find_all('loc')]
    # WordPressの場合、末尾に"/"が付いていないと不正なURLも混じるかもしれないので適宜絞り込み
    return [url for url in urls if url.endswith('/')]

# ---------------------------------------
# 3) 不要部分を削除する関数 (例)
# ---------------------------------------
def remove_footer(text):
    # フッターや不要部分を正規表現で削除
    footer_pattern = r"(検索:.*?All Rights Reserved\.)"
    return re.sub(footer_pattern, '', text, flags=re.DOTALL).strip()

def remove_top_bar(text):
    top_bar_pattern = r"(神社仏閣オンライン.*?お問合せ/会社概要)"
    return re.sub(top_bar_pattern, '', text, flags=re.DOTALL).strip()

def remove_title(text):
    title_pattern = r"(: 神社仏閣オンライン)"
    return re.sub(title_pattern, '', text, flags=re.DOTALL).strip()

# ---------------------------------------
# 4) 要約を行う関数
# ---------------------------------------
def summarize_text(text):
    """
    ChatCompletionを使ってテキストを要約。
    長いテキストに対しては短めの分割処理やトークン数注意が必要。
    """
    prompt = f"""
    次の文章を要約してください。重要なポイントだけを残して簡潔にまとめてください。

    文章:
    {text}
    """
    response = openai.ChatCompletion.create(
        model=GPT_MODEL,
        messages=[
            {"role": "system", "content": "あなたは文章要約の専門家です。"},
            {"role": "user", "content": prompt}
        ],
        temperature=0.5,
    )
    return response.choices[0].message.content.strip()

# ---------------------------------------
# 5) ページをスクレイピングして要約CSVを作る
# ---------------------------------------
def create_summarized_csv(sitemap_url, csv_file):
    urls = fetch_sitemap_urls(sitemap_url)
    page_contents = []

    for url in urls:
        try:
            page_response = requests.get(url)
            page_soup = BeautifulSoup(page_response.content, 'html.parser')

            # タイトル
            title_tag = page_soup.find('title')
            title = title_tag.text if title_tag else 'No Title'
            title = remove_title(title)

            # カテゴリ/タグ（rel="category tag" のリンク）
            category_tags = page_soup.find_all('a', rel='category tag')
            tag_list = [link.text for link in category_tags]
            tag_str = ', '.join(tag_list) if tag_list else 'No Tag'

            # 本文 (今回は get_text() でまるごと抜く簡易例)
            body = page_soup.get_text(separator=" ", strip=True)
            body = remove_top_bar(body)
            body = remove_footer(body)

            # 必要に応じて文字数制限。長文は先頭部だけ要約などの工夫が必要。
            # 例: Bodyが極端に長いとトークン数オーバーするため、先頭2000文字だけ要約
            excerpt = body[:2000] if len(body) > 2000 else body

            if len(excerpt) > 200:
                summary = summarize_text(excerpt)
            else:
                summary = excerpt

            # データ保存
            page_contents.append({
                'url': url,
                'title': title,
                'tag': tag_str,
                'body': summary
            })

            print(f"[{url}] -> 要約完了")
            time.sleep(1)  # API連打防止
        except Exception as e:
            print(f"[{url}] -> エラー: {e}")

    # DataFrame化・CSV保存
    df = pd.DataFrame(page_contents)
    df.to_csv(csv_file, index=False, encoding='utf-8')
    print(f"==== CSV保存完了: {csv_file} ====")

# ---------------------------------------
# 6) CSVを読み込み、埋め込みベクトルを生成して BINファイルに保存
# ---------------------------------------
def generate_embeddings_and_save(csv_file, bin_file):
    df = pd.read_csv(csv_file)

    # OpenAI Embedding API で埋め込みベクトルを取得する関数
    def get_embedding(text):
        try:
            emb_response = openai.Embedding.create(
                model=EMBEDDING_MODEL,
                input=text
            )
            return emb_response['data'][0]['embedding']
        except Exception as e:
            print(f"Embeddingエラー: {e}")
            return [0.0] * 1536  # 次元数はモデルにより要変更

    # 埋め込み生成
    all_embeddings = []
    for i, row in df.iterrows():
        body_text = row['body']
        embedding = get_embedding(body_text)
        all_embeddings.append(embedding)
        # 適宜ウェイトを置く
        time.sleep(0.5)

    # numpy配列化
    embeddings_array = np.array(all_embeddings, dtype=np.float32)
    embeddings_array.tofile(bin_file)
    print(f"==== BIN保存完了: {bin_file} ====")

    # 次回以降の検索を簡易にするため、DataFrameに埋め込みも保持しておきたい場合は
    # JSON等にシリアライズするなどの手段で保存してもよい

# ---------------------------------------
# 7) BIN読み込み→類似度検索→ChatGPT回答 (インタラクティブ例)
# ---------------------------------------
def interactive_qa(csv_file, bin_file):
    df = pd.read_csv(csv_file)
    
    # モデルの埋め込み次元 (text-embedding-3-small は不明、ada-002は1536)
    EMBED_DIM = 1536

    # BIN読み込み
    bin_data = np.fromfile(bin_file, dtype=np.float32)
    num_records = len(df)
    if bin_data.size != num_records * EMBED_DIM:
        raise ValueError("BINファイルのサイズがCSV行数×次元数と一致しません。")

    # 2次元にリシェイプ
    embeddings_matrix = bin_data.reshape((num_records, EMBED_DIM))
    
    # DataFrameにembedding列として取り込む
    df["embedding"] = embeddings_matrix.tolist()

    # 検索用関数
    def search_query(query, top_n=3):
        # クエリをベクトル化
        try:
            query_emb = openai.Embedding.create(
                model=EMBEDDING_MODEL,
                input=query
            )['data'][0]['embedding']
        except Exception as e:
            print(f"クエリ埋め込みエラー: {e}")
            query_emb = [0.0]*EMBED_DIM
        
        # コサイン類似度
        query_emb_np = np.array(query_emb, dtype=np.float32).reshape(1, -1)
        all_embs_np = np.array(df["embedding"].tolist(), dtype=np.float32)
        sims = cosine_similarity(query_emb_np, all_embs_np)[0]

        df["similarity"] = sims
        top_results = df.sort_values(by="similarity", ascending=False).head(top_n)
        return top_results[["url", "title", "tag", "body", "similarity"]]

    # ChatGPTへ問い合わせ
    def generate_answer(query):
        results = search_query(query, top_n=3)

        # コンテキスト
        context = "\n\n".join(results["body"].tolist())
        urls = "\n".join(results["url"].tolist())

        prompt = f"""
        以下の情報を元に、質問に答えてください:

        神社、仏閣の知識:
        {context}

        質問:
        {query}

        回答:
        """

        try:
            response = openai.ChatCompletion.create(
                model=GPT_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "あなたは神社、仏閣について知識を持っています。"
                            "知識に基づいて具体的な神社、仏閣の名前を出し300文字以内で答えて下さい。"
                        )
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.5,
            )
            answer = response.choices[0].message.content.strip()
        except Exception as e:
            print(f"ChatCompletionエラー: {e}")
            answer = "回答を生成できませんでした。"
        
        return answer, urls

    # ループでQ&A
    while True:
        user_input = input("質問を入力してください (終了するには 'exit'): ")
        if user_input.lower() == 'exit':
            print("終了します。")
            break

        ans, refs = generate_answer(user_input)
        print("\n=== 回答 ===")
        print(ans)
        print("\n=== おすすめの記事 ===")
        print(refs)
        print("\n")

# ---------------------------------------
# メイン実行
# ---------------------------------------
if __name__ == "__main__":

    # (A) CSVファイルをまだ作っていない場合 → サイトマップからスクレイピング & 要約
    if not os.path.exists(CSV_FILE):
        create_summarized_csv(SITEMAP_URL, CSV_FILE)
    else:
        print(f"既に {CSV_FILE} が存在します。")

    # (B) BINファイルがない場合は埋め込み生成→保存
    if not os.path.exists(BIN_FILE):
        generate_embeddings_and_save(CSV_FILE, BIN_FILE)
    else:
        print(f"既に {BIN_FILE} が存在します。")

    # (C) インタラクティブにQ&Aテスト
    interactive_qa(CSV_FILE, BIN_FILE)