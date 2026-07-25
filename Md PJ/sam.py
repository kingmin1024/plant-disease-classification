import os
import torch
import json

# 경로 설정
LATEST_FILE = "/home/ubuntu/.ollama/models/manifests/registry.ollama.ai/library/llama3.3/latest"
BLOB_DIR = "/home/ubuntu/.ollama/models/blobs"
OUTPUT_DIR = "/home/ubuntu/project/models/huggingface_compatible"

# GPU 디바이스 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# .latest 파일 로드
def load_latest_metadata(latest_file):
    """
    Load the .latest file to extract metadata and blob references.
    """
    with open(latest_file, "r", encoding="utf-8") as f:
        return json.load(f)

# Blob 파일 매핑
def map_blob_files(metadata, blob_dir):
    """
    Map sha256 blobs based on the metadata from .latest file.
    """
    layers = metadata.get("layers", [])
    blob_paths = []

    for layer in layers:
        digest = layer.get("digest", "").replace("sha256:", "")
        blob_path = os.path.join(blob_dir, f"sha256-{digest}")
        if os.path.exists(blob_path):
            blob_paths.append((blob_path, layer.get("mediaType", "unknown")))
        else:
            print(f"Missing blob file: {blob_path}")

    return blob_paths

# Blob 파일 데이터를 조각 단위로 로드 (진행률 표시 포함)
def load_blob_in_chunks_with_progress(blob_path, chunk_size=8 * 1024 * 1024):  # 8MB 단위
    """
    Read a large blob file in chunks and display progress.
    """
    file_size = os.path.getsize(blob_path)
    read_size = 0

    with open(blob_path, "rb") as f:
        while chunk := f.read(chunk_size):
            read_size += len(chunk)
            progress = (read_size / file_size) * 100
            print(f"\rReading {blob_path}: {progress:.2f}% completed", end="")
            yield chunk

    print()  # 줄바꿈

# 개별 Blob 파일 처리
def process_blob(blob_file, media_type, output_dir):
    """
    Process a single Blob file and save its tensor representation using GPU.
    """
    if "model" in media_type or "params" in media_type:
        output_path = os.path.join(output_dir, f"{os.path.basename(blob_file)}.pt")

        with open(output_path, "wb") as out_file:
            for chunk in load_blob_in_chunks_with_progress(blob_file):
                # GPU에서 텐서 생성 후 디스크에 바로 저장
                tensor = torch.tensor(bytearray(chunk), dtype=torch.uint8, device=device)
                torch.save(tensor, out_file)  # 즉시 저장
                torch.cuda.empty_cache()  # GPU 메모리 캐시 비우기

        print(f"Saved: {output_path}")
    else:
        print(f"Skipping non-model blob: {blob_file} ({media_type})")

# Blob 데이터를 순차적으로 처리
def convert_to_huggingface_format(blob_files, output_dir):
    """
    Process Blob files sequentially and save as Hugging Face compatible format.
    """
    os.makedirs(output_dir, exist_ok=True)

    for idx, (blob_file, media_type) in enumerate(blob_files):
        print(f"Processing Blob {idx + 1}/{len(blob_files)}: {blob_file} ({media_type})")
        process_blob(blob_file, media_type, output_dir)

    print(f"All blobs processed and saved to {output_dir}.")

# 실행
if __name__ == "__main__":
    metadata = load_latest_metadata(LATEST_FILE)
    blob_files = map_blob_files(metadata, BLOB_DIR)

    if blob_files:
        convert_to_huggingface_format(blob_files, OUTPUT_DIR)
    else:
        print("No Blob files found for conversion.")
