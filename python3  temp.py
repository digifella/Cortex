 python3 << 'EOF'
  import os
  os.environ["HF_HUB_OFFLINE"] = "0"
  os.environ["TRANSFORMERS_OFFLINE"] = "0"

  from sentence_transformers import SentenceTransformer

  print("Downloading nvidia/NV-Embed-v2...")
  model = SentenceTransformer("nvidia/NV-Embed-v2")
  print("✅ Model downloaded and cached successfully")

  # Test it
  test_emb = model.encode("test")
  print(f"✅ Model working - embedding dimension: {len(test_emb)}")
  EOF