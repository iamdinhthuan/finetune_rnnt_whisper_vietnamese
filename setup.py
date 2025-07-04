from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="whisper-rnnt",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Vietnamese Speech Recognition with Whisper Encoder and RNN-T Decoder",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/whisper_rnnt",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Multimedia :: Sound/Audio :: Speech",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "isort>=5.12.0",
        ],
        "ui": [
            "gradio>=3.40.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "whisper-rnnt-train=train:main",
            "whisper-rnnt-demo=infer_stream_ui:main",
        ],
    },
    include_package_data=True,
    package_data={
        "utils": ["*.npz", "tokenizer_spe_bpe_v1024_pad/*"],
    },
)
