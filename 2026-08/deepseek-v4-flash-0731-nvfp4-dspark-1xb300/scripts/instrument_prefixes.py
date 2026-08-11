P = "/usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v4/quant_config.py"
s = open(P).read()
anchor = "    def get_quant_method(self, layer, prefix):"
inject = anchor + """
        try:
            open("/tmp/prefixes.txt", "a").write(
                type(layer).__name__ + " | " + repr(prefix) + "\\n")
        except Exception:
            pass"""
assert anchor in s, "ANCHOR_NOT_FOUND"
s = s.replace(anchor, inject, 1)
open(P, "w").write(s)
print("instrumented")
