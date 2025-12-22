import os
import mlx.core as mx
from mlx_lm import load

def setup_and_chat():
    # --- 1. 設定外接硬碟路徑 ---
    # 請將 '你的硬碟名稱' 替換為你在 Finder 看到的名稱
    disk_name = "T7_SSD" 
    external_path = f"/Volumes/{disk_name}/huggingface_cache"
    
    # 檢查外接硬碟是否已掛載
    if not os.path.exists(f"/Volumes/{disk_name}"):
        print(f"❌ 錯誤：找不到外接硬碟 '{disk_name}'，請確認是否已插入並正確掛載。")
        return

    # 建立快取資料夾
    os.makedirs(external_path, exist_ok=True)
    
    # 設定 Hugging Face 環境變數，讓所有下載都去外接硬碟
    os.environ["HF_HOME"] = external_path
    print(f"✅ 已將模型路徑設定為：{external_path}")

    # --- 2. 選擇模型 ---
    # 使用 MLX 格式的 4-bit 量化模型（更適合 MacBook Air）
    model_id = "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit"
    
    # 如果本地已有模型，可以使用本地路徑
    # local_model_path = "/Volumes/T7_SSD/Qwen2.5-Coder-7B-Instruct-4bit"
    # if os.path.exists(local_model_path):
    #     model_id = local_model_path

    print(f"正在載入模型 {model_id}... (第一次執行會自動下載至外接硬碟)")
    print("   ℹ️  使用 MLX 格式（專為 Apple Silicon 優化，4-bit 量化）")

    # --- 3. 載入模型與 Tokenizer（MLX 方式）---
    print("   正在載入模型和 Tokenizer...")
    print("   ⚠️  注意：MLX 自動使用 Apple Silicon GPU，無需手動設定設備")
    
    try:
        # MLX 的 load 函數會同時載入模型和 tokenizer
        model, tokenizer = load(model_id)
        print("✅ 模型載入完成！")
        print("   ✓ 使用 MLX 框架（Apple Silicon 原生加速）")
        print("   ✓ 4-bit 量化版本（記憶體占用約 4GB）")
    except Exception as e:
        print(f"❌ 載入模型失敗：{e}")
        print("   請確認模型 ID 正確，或檢查網路連線")
        return

    # --- 4. 進行簡單問答 ---
    print("\n--- 模型載入完成！你可以開始提問了 (輸入 'exit' 退出) ---")
    
    while True:
        user_input = input("你：")
        if user_input.lower() in ['exit', 'quit', '離開']:
            break

        # 設定對話格式 (Qwen 使用 ChatML 格式)
        messages = [
            {"role": "system", "content": "你是一個專業的程式設計助手，擅長程式碼生成和問題解答。請用繁體中文回答。"},
            {"role": "user", "content": user_input},
        ]

        # 使用 tokenizer 的 apply_chat_template 來格式化對話（確保格式正確）
        print("   [1/2] 正在處理輸入...")
        try:
            # 使用 tokenizer 格式化對話
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,  # 返回字串而不是 token IDs
                add_generation_prompt=True
            )
        except Exception as e:
            # 如果 apply_chat_template 不支援，使用手動格式
            print(f"   ⚠️  使用手動格式（apply_chat_template 失敗：{e}）")
            prompt = f"<|im_start|>system\n{messages[0]['content']}<|im_end|>\n<|im_start|>user\n{user_input}<|im_end|>\n<|im_start|>assistant\n"

        print("   [2/2] 正在生成回答（使用 MLX，Apple Silicon 加速，流式輸出）...")
        print("AI：", end="", flush=True)  # 開始輸出，不換行

        # 手動實作流式生成循環
        try:
            max_tokens = 512
            
            # 將 prompt 編碼為 tokens
            tokens = tokenizer.encode(prompt)
            tokens = mx.array(tokens)
            
            # 用於儲存生成的 tokens（用於流式輸出）
            generated_tokens = []
            last_decoded_length = 0  # 追蹤已解碼的長度，用於增量解碼
            
            # 生成循環
            for step in range(max_tokens):
                # 前向傳播，獲取 logits
                logits = model(tokens[None, :])
                logits = logits[0, -1, :]  # 取最後一個位置的 logits
                
                # 使用貪婪解碼（選擇機率最高的 token）
                next_token = mx.argmax(logits)
                next_token = int(next_token.item())
                
                # 檢查是否遇到結束符
                if next_token == tokenizer.eos_token_id:
                    break
                
                # 將新 token 加入序列
                generated_tokens.append(next_token)
                tokens = mx.concatenate([tokens, mx.array([next_token])])
                
                # 流式解碼：每次解碼整個序列，只輸出新增的部分
                try:
                    full_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
                    # 只輸出新增的文字部分
                    new_text = full_text[last_decoded_length:]
                    if new_text:  # 如果有新文字才輸出
                        print(new_text, end="", flush=True)
                        last_decoded_length = len(full_text)
                except Exception:
                    # 如果解碼失敗，跳過這次輸出
                    pass
            
            print("\n   ✓ 生成完成！\n")
        except Exception as e:
            print(f"\n   ❌ 生成時發生錯誤：{e}")
            print("   請重試或檢查記憶體是否充足\n")
            continue

        print("-" * 30)

if __name__ == "__main__":
    setup_and_chat()