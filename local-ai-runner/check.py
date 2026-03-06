from llama_cpp import Llama
import traceback
p = "/Volumes/PortableSSD/TMP/Qwen3.5-0.8B-UD-Q2_K_XL.gguf"
try:
    m = Llama(model_path=p)
    print("OK")
except Exception:
    traceback.print_exc()