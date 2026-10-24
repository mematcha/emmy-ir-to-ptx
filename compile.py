import sys
from pathlib import Path

ir_path = Path(sys.argv[1])
name = ir_path.name
if name.endswith(".ir.json"):
    name = name[: -len(".ir.json")]
elif name.endswith(".json"):
    name = name[: -len(".json")]
ptx_path = ir_path.with_name(name + ".ptx")
ptx_path.write_text("")
print(f"status: compiler not implemented; created empty {ptx_path}")
