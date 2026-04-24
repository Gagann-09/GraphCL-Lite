import json, ast, sys

nb = json.load(open("GraphCL_Lite_GNN_MiniProject.ipynb", "r", encoding="utf-8"))
errors = 0
for i, cell in enumerate(nb["cells"]):
    if cell["cell_type"] == "code":
        code = "\n".join(cell["source"])
        # Skip cells that start with ! (shell commands)
        code_stripped = code.strip()
        if code_stripped.startswith("# Install") or code_stripped.startswith("!"):
            print(f"  Cell {i+1:2d} [code]: SKIPPED (install cell)")
            continue
        try:
            ast.parse(code)
            print(f"  Cell {i+1:2d} [code]: SYNTAX OK")
        except SyntaxError as e:
            print(f"  Cell {i+1:2d} [code]: SYNTAX ERROR - {e}")
            errors += 1
    else:
        lines = len(cell["source"])
        print(f"  Cell {i+1:2d} [md]  : {lines} lines")

print(f"\nResult: {errors} syntax errors found")
sys.exit(errors)
