<?php

// REST APIエンドポイント登録
add_action('rest_api_init', function () {
    register_rest_route('openai/v1', '/chat', array(
        'methods' => 'POST',
        'callback' => 'handle_openai_chat',
        'permission_callback' => '__return_true', // 認証不要
    ));
});

// OpenAI APIキーの読み込み (wp-config.php に定義)
define('OPENAI_API_KEY', defined('OPENAI_API_KEY') ? OPENAI_API_KEY : null);

// 埋め込みの次元数 (例: 1536)
define('EMBED_DIMENSION', 1536);

// データファイルパス
define('CSV_FILE', ABSPATH . 'wp-content/plugins/openai-rag/data/summarized_content.csv');
define('BIN_FILE', ABSPATH . 'wp-content/plugins/openai-rag/data/embeddings.bin');

// データ読み込み関数
function load_data() {
    $csv_file = CSV_FILE;
    $bin_file = BIN_FILE;

    // CSVデータ読み込み
    $data = [];
    if (($handle = fopen($csv_file, 'r')) !== false) {
        $headers = fgetcsv($handle);
        while (($row = fgetcsv($handle)) !== false) {
            $data[] = array_combine($headers, $row);
        }
        fclose($handle);
    }

    // バイナリ (.bin) 読み込み
    $bin_data = file_get_contents($bin_file);
    // float32配列としてunpack → EMBED_DIMENSION(1536)次元ずつに分割
    $embeddings_array = unpack('f*', $bin_data);
    $embeddings = array_chunk($embeddings_array, EMBED_DIMENSION);

    // CSVデータに埋め込みベクトルを追加
    foreach ($data as $index => &$row) {
        $row['embedding'] = $embeddings[$index];
    }

    return $data;
}

// コサイン類似度計算
function cosine_similarity($vec1, $vec2) {
    $dot_product = 0.0;
    foreach ($vec1 as $i => $val1) {
        $dot_product += $val1 * $vec2[$i];
    }

    $magnitude1 = 0.0;
    $magnitude2 = 0.0;
    foreach ($vec1 as $v1) {
        $magnitude1 += $v1 * $v1;
    }
    foreach ($vec2 as $v2) {
        $magnitude2 += $v2 * $v2;
    }
    $magnitude1 = sqrt($magnitude1);
    $magnitude2 = sqrt($magnitude2);

    if ($magnitude1 == 0.0 || $magnitude2 == 0.0) {
        // ゼロベクトル対策
        return 0.0;
    }

    return $dot_product / ($magnitude1 * $magnitude2);
}

// 類似度検索
function search_query($query, $data, $top_n = 3) {
    // OpenAI埋め込みモデルでクエリをベクトル化
    $api_key = OPENAI_API_KEY;
    $response = wp_remote_post('https://api.openai.com/v1/embeddings', [
        'headers' => [
            'Content-Type' => 'application/json',
            'Authorization' => 'Bearer ' . $api_key,
        ],
        'body' => json_encode([
            'model' => 'text-embedding-ada-002', // 例: 適宜モデル名を指定
            'input' => $query
        ]),
    ]);

    if (is_wp_error($response)) {
        return [];
    }

    $body = json_decode(wp_remote_retrieve_body($response), true);
    $query_embedding = $body['data'][0]['embedding'] ?? null;
    if (!$query_embedding || !is_array($query_embedding)) {
        // エラー時は空を返す
        return [
            [
                'url'   => '',
                'title' => '',
                'body'  => '',
                'similarity' => 0.0
            ]
        ];
    }

    // データとの類似度を計算
    $similarities = [];
    foreach ($data as $row) {
        $similarity = cosine_similarity($query_embedding, $row['embedding']);
        $similarities[] = [
            'url'       => $row['url'],
            'title'     => $row['title'],
            'body'      => $row['body'],
            'similarity'=> $similarity
        ];
    }

    // 類似度順ソート → 上位N件取得
    usort($similarities, fn($a, $b) => $b['similarity'] <=> $a['similarity']);
    return array_slice($similarities, 0, $top_n);
}

// OpenAI APIリクエスト処理(RAG統合)
function handle_openai_chat(WP_REST_Request $request) {
    $message = $request->get_param('message');

    // メッセージが未入力の場合
    if (!$message) {
        return new WP_REST_Response(['reply' => 'メッセージがありません。'], 400);
    }

    // APIキーが存在しない場合
    if (!OPENAI_API_KEY) {
        return new WP_REST_Response(['reply' => 'APIキーが設定されていません。'], 500);
    }

    // CSVとBINのデータを読み込み
    $data = load_data();
    if (!$data) {
        return new WP_REST_Response(['reply' => 'データがロードできませんでした'], 500);
    }

    // 類似度検索
    $results = search_query($message, $data, 3);

    // コンテキスト文章を作成
    $context = implode("\n\n", array_column($results, 'body'));
    $references = implode("\n", array_column($results, 'url'));
    $similarities = implode("\n", array_column($results, 'similarity'));

    // OpenAI Chat APIへコンテキストを付与して問い合わせ
    $endpoint = 'https://api.openai.com/v1/chat/completions';
    $api_key = OPENAI_API_KEY;

    // 実際にシステムに与えるプロンプト例
    $prompt = "以下の情報を元に、質問に答えてください: \n\n知識:\n" . $context . "\n\n質問:\n" . $message . "\n\n回答:";

    $post_data = [
        'model' => 'gpt-3.5-turbo',
        'messages' => [
            [
                'role' => 'system',
                'content' => 'あなたは神社、仏閣について知識を持っています。知識に基づいて具体的な神社、仏閣の名前を出し300文字以内で答えて下さい。'
            ],
            [
                'role' => 'user',
                'content' => $prompt
            ]
        ],
        'max_tokens' => 3000,
    ];

    $response = wp_remote_post($endpoint, [
        'headers' => [
            'Content-Type' => 'application/json',
            'Authorization' => 'Bearer ' . $api_key,
        ],
        'body' => json_encode($post_data, JSON_UNESCAPED_UNICODE),
        'timeout' => 30,
    ]);

    if (is_wp_error($response)) {
        return new WP_REST_Response(['reply' => 'サーバーエラー: リクエスト失敗'], 500);
    }

    $body = json_decode(wp_remote_retrieve_body($response), true);
    $reply = $body['choices'][0]['message']['content'] ?? '応答がありません。';

    // APIレスポンスを返す
    return new WP_REST_Response([
        'reply'      => $reply,
        'context'    => $context,
        'references' => $references,
        'similarity' => $similarities,
        'dimension'  => EMBED_DIMENSION,
    ], 200);
}