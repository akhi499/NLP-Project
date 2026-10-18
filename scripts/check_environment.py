import importlib.metadata as m, sys
def ver(name):
    try:return m.version(name)
    except m.PackageNotFoundError:return "not installed"
print(f"Python: {sys.version.split()[0]}")
for name in ["torch","transformers","scikit-learn","pandas","streamlit","PyYAML"]: print(f"{name}: {ver(name)}")
try:
 import torch
 print(f"CUDA available: {torch.cuda.is_available()}")
 if torch.cuda.is_available():
  p=torch.cuda.get_device_properties(0)
  print(f"GPU: {p.name}")
  print(f"VRAM: {p.total_memory/1024**3:.2f} GiB")
  x=torch.randn((256,256),device="cuda",requires_grad=True); (x@x).mean().backward()
  print("CUDA forward/backward smoke check: passed")
except Exception as e:print(f"PyTorch runtime check failed: {e}")

