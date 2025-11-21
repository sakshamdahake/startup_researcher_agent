# scripts/check_langchain_imports.py
"""
Quick helper to check which LangChain import paths work in your environment.

Run:
    python -m scripts.check_langchain_imports

It will print a short report you can use to update app/services/agent_runner.py
with the single working import you prefer.
"""

import importlib
import sys
import traceback

CANDIDATES = [
    # modern v1
    ("langchain.agents", "create_agent"),
    ("langchain_core.agents", "create_agent"),
    ("langchain.agents", "create_react_agent"),
    ("langchain_core.agents", "create_react_agent"),

    # callbacks (common v1 paths)
    ("langchain.callbacks.manager", "CallbackManager"),
    ("langchain.callbacks.base", "BaseCallbackHandler"),
    ("langchain_core.callbacks.manager", "CallbackManager"),
    ("langchain_core.callbacks.base", "BaseCallbackHandler"),

    # older / alternative
    ("langchain.agents", "create_llm_agent"),
]

def try_import(module_name: str, attr: str):
    try:
        mod = importlib.import_module(module_name)
        obj = getattr(mod, attr, None)
        if obj is not None:
            return True, obj
        else:
            return False, f"module '{module_name}' found but attribute '{attr}' missing"
    except Exception as e:
        return False, f"import error: {type(e).__name__}: {e}"

def main():
    results = []
    print("Checking LangChain import candidates...\n")
    for module_name, attr in CANDIDATES:
        ok, res = try_import(module_name, attr)
        if ok:
            print(f"[OK]   {module_name}.{attr}  -> {res}")
            results.append((module_name, attr, True, res))
        else:
            print(f"[FAIL] {module_name}.{attr}  -> {res}")
            results.append((module_name, attr, False, res))

    print("\nSummary - usable imports (first column shows module, second the attribute):\n")
    usable = [(m, a, obj) for (m, a, ok, obj) in results if ok]
    if not usable:
        print("No usable imports found. Full trace of last failures printed below.\n")
        # Print full traceback for debugging (only if nothing worked)
        for (m, a, ok, obj) in results:
            if not ok:
                print(f" - {m}.{a}: {obj}")
        sys.exit(2)

    # print the best candidate (prefer create_agent)
    preferred = None
    for m, a, obj in usable:
        if a == "create_agent":
            preferred = (m, a, obj)
            break
    if preferred is None:
        preferred = usable[0]

    print("Recommended import to use in your code:")
    print(f"  from {preferred[0]} import {preferred[1]}\n")
    print("You can copy-paste that line into agent_runner.py or use it to adapt the runner.")
    print("\nOptional: try a tiny smoke call for CallbackManager / BaseCallbackHandler if available.")

    # If CallbackManager + BaseCallbackHandler both available, show them
    cbm = None
    bch = None
    for m, a, ok, obj in results:
        if ok and a == "CallbackManager":
            cbm = obj
        if ok and a == "BaseCallbackHandler":
            bch = obj

    if cbm and bch:
        try:
            print("\nAttempting to instantiate CallbackManager and BaseCallbackHandler (small smoke test)...")
            class TinyHandler(bch):
                pass
            cm = cbm([TinyHandler()])
            print(" - CallbackManager instantiated OK:", type(cm))
            print(" - TinyHandler subclass OK:", TinyHandler)
        except Exception as e:
            print(" - Smoke test failed:", type(e).__name__, e)
            traceback.print_exc()

if __name__ == "__main__":
    main()