import subprocess

# Cập nhật pip lên phiên bản mới nhất
subprocess.run(["pip", "install", "--upgrade", "pip"])

# Cài đặt các gói từ tệp requirements.txt
subprocess.run(["pip", "install", "-r", "requirements.txt"])

# Cài đặt các gói PyTorch với URL index cụ thể
subprocess.run([
    "pip", "install",
    "torch==2.5.1",
    "torchvision==0.20.1",
    "torchaudio==2.5.1",
    "--index-url", "https://download.pytorch.org/whl/cu118"
])