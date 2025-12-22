import os
from mlx_lm import load, generate

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

        print("   [2/2] 正在生成回答（使用 MLX，Apple Silicon 加速）...")

        # 使用 MLX 的 generate 函數生成回答
        try:
            # MLX 的 generate 函數會自動處理 tokenization 和生成
            # 注意：MLX-LM 的 generate 函數不支援 temperature、top_p 等參數
            # 只使用基本參數：prompt 和 max_tokens
            response = generate(
                model,
                tokenizer,
                prompt=prompt,
                max_tokens=512,  # 最大生成 token 數
                verbose=False  # 不顯示詳細進度
            )
            print("   ✓ 生成完成！\n")
        except Exception as e:
            print(f"   ❌ 生成時發生錯誤：{e}")
            print("   請重試或檢查記憶體是否充足\n")
            continue

        # MLX 的 generate 直接返回字串，不需要 decode
        print(f"AI：{response}")
        print("-" * 30)

if __name__ == "__main__":
    setup_and_chat()