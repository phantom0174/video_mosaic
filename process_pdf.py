import os
import fitz  # PyMuPDF
from PIL import Image
import io
import multiprocessing
from multiprocessing import Manager
from tqdm import tqdm
import time

# 設定區 ==========================================
INPUT_FOLDER = r'epstein'  # PDF 資料夾路徑
OUTPUT_FOLDER = r'epstein_frames'   # 圖片輸出路徑
MIN_IMAGE_SIZE = 0  # 忽略小於 2KB 的圖片 (過濾 icon 或分隔線)
# ===============================================

def worker_process(file_args):
    """
    單一 PDF 處理函數，被進程池調用
    """
    pdf_path, output_dir, counter, lock = file_args
    
    extracted_count = 0
    try:
        # 使用 PyMuPDF 打開 PDF (支援 lazy loading，記憶體友善)
        doc = fitz.open(pdf_path)
        
        for page_index in range(len(doc)):
            page = doc[page_index]
            image_list = page.get_images(full=True)
            
            for img_index, img in enumerate(image_list):
                xref = img[0]
                
                # 取得圖片原始數據
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                
                # 過濾過小的圖片 (雜訊、Icon)
                if len(image_bytes) < MIN_IMAGE_SIZE:
                    continue

                try:
                    # 使用 Pillow 讀取圖片
                    image = Image.open(io.BytesIO(image_bytes))
                    
                    # === 轉為灰階 ===
                    gray_image = image.convert('L')
                    
                    # === 安全獲取流水號 ===
                    with lock:
                        current_id = counter.value
                        counter.value += 1
                    
                    # 建立檔名 (例如: 00000123.jpg)
                    filename = f"{current_id:08d}.jpg"
                    save_path = os.path.join(output_dir, filename)
                    
                    # 儲存圖片 (JPEG 格式)
                    gray_image.save(save_path, "JPEG", quality=85)
                    extracted_count += 1
                    
                except Exception as e_img:
                    # 某些圖片格式可能損壞，跳過不處理，避免中斷整個流程
                    continue
                    
        doc.close()
        return extracted_count
        
    except Exception as e:
        print(f"Error processing {pdf_path}: {e}")
        return 0

def main():
    # 1. 準備輸出資料夾
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
        
    # 2. 獲取所有 PDF 檔案列表
    pdf_files = [
        os.path.join(INPUT_FOLDER, f) 
        for f in os.listdir(INPUT_FOLDER) 
        if f.lower().endswith('.pdf')
    ]
    
    print(f"找到 {len(pdf_files)} 個 PDF 檔案，準備處理...")
    
    # 3. 設定多進程
    # 使用 Manager 來管理跨進程的變數
    manager = Manager()
    shared_counter = manager.Value('i', 1)  # 從 1 開始計數
    shared_lock = manager.Lock()            # 鎖，防止流水號重複
    
    # 獲取 CPU 核心數，保留 1 核心給系統
    num_processes = max(1, multiprocessing.cpu_count() - 1)
    print(f"啟動 {num_processes} 個並行進程...")

    # 4. 準備參數 (將參數打包傳給 worker)
    tasks = []
    for pdf in pdf_files:
        tasks.append((pdf, OUTPUT_FOLDER, shared_counter, shared_lock))

    # 5. 開始執行並顯示進度條
    total_extracted = 0
    start_time = time.time()
    
    with multiprocessing.Pool(processes=num_processes) as pool:
        # 使用 tqdm 顯示進度，imap_unordered 讓完成的任務先回報
        for count in tqdm(pool.imap_unordered(worker_process, tasks), total=len(tasks)):
            total_extracted += count

    end_time = time.time()
    duration = end_time - start_time
    
    print("-" * 30)
    print("處理完成！")
    print(f"總耗時: {duration:.2f} 秒")
    print(f"共處理 {len(pdf_files)} 個 PDF")
    print(f"共擷取 {total_extracted} 張圖片")
    print(f"圖片儲存於: {OUTPUT_FOLDER}")

if __name__ == '__main__':
    # Windows 下必須將執行邏輯放在 if __name__ == '__main__': 區塊內
    main()