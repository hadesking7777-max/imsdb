"""Offline bilingual (English/Chinese) chatbot over the local IMSDb knowledge base.

Everything runs on this machine: a local GGUF chat model and embedding model (llama-cpp-python), MongoDB for
structured facts and keyword search, and a NumPy vector index on disk. No network access at run time.
"""
