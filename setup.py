"""
MiniChat (MiniLM) - Local-first LLM Desktop Application
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="minichat",
    version="1.0.0",
    author="MiniChat Team",
    description="A local-first LLM desktop application with RAG and custom agents",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "streamlit>=1.28.0",
        "ollama>=0.1.0",
        "chromadb>=0.4.0",
        "sentence-transformers>=2.2.0",
        "langchain>=0.1.0",
        "langchain-community>=0.0.10",
        "pypdf>=3.0.0",
        "python-docx>=0.8.11",
        "pywebview>=4.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "hypothesis>=6.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "minichat=miniLM.src.desktop:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
