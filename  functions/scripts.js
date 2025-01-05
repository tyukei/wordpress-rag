document.addEventListener("DOMContentLoaded", function () {
    const button = document.getElementById("sendMessage");
    const input = document.getElementById("userInput");
    const output = document.getElementById("responseOutput");
  
    button.addEventListener("click", async () => {
      const userInput = input.value.trim();
      if (!userInput) {
        output.textContent = "メッセージを入力してください。";
        return;
      }
  
      // 「生成中...」表示
      output.textContent = "生成中...";
  
      try {
        const response = await fetch("/wp-json/openai/v1/chat", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            message: userInput,
          }),
        });
  
        if (!response.ok) {
          output.textContent = `エラーが発生しました。ステータスコード: ${response.status}`;
          return;
        }
  
        const data = await response.json();
  
        // 応答とリンクの表示
        let outputText = `<p>${data.reply}</p>`; // 応答メッセージ
        outputText += "<p>あなたにおすすめの記事:</p><ul>";
  
        // リンク一覧
        const links = data.references.split("\n");
        links.forEach(link => {
          if (link.trim()) { // 空文字対策
            outputText += `<li><a href="${link}" target="_blank">${link}</a></li>`;
          }
        });
        outputText += "</ul>";
  
        output.innerHTML = outputText;
      } catch (error) {
        output.textContent = "通信エラーが発生しました。";
        console.error("エラー詳細:", error);
      }
    });
  });