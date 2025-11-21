# scripts/run_agent_cli.py
"""
Simple CLI to run the LangChain agent via app.services.agent_runner.create_agent_runner
Usage:
    python -m scripts.run_agent_cli --query "your query"
"""

from app.services.agent_runner import create_agent_runner, run_agent

def main():
    agent = create_agent_runner()    # <- new factory name
    query = "Startup idea: subscription meal kits for busy parents"
    print("Running agent for:", query)
    out = run_agent(agent, query, verbose=True)
    print("\n=== FINAL TEXT ===\n")
    print(out["text"][:2000])
    print("\n=== TRACE ===\n")
    for ev in out.get("trace", []):
        print(ev)

if __name__ == "__main__":
    main()

# from app.services.agent_runner import create_agent_runner
# agent = create_agent_runner()
# # try common places LangChain stores tools
# print("agent attr keys:", list(vars(agent).keys())[:40])
# print("agent.tools?", getattr(agent, "tools", None))
# print("agent._tracer:", getattr(agent, "_tracer", None))
# # If agent has an executor or toolkit, inspect it:
# for name in ("executor", "agent_executor", "toolkit", "tools"):
#     obj = getattr(agent, name, None)
#     if obj is not None:
#         print(f"{name} ->", obj)
#         try:
#             print("  element names:", [getattr(t, "__name__", getattr(t,"name", str(t))) for t in (obj if isinstance(obj, (list,tuple)) else getattr(obj,"tools", getattr(obj,"_tools", [])))][:20])
#         except Exception:
#             pass


# from app.services.agent_runner import create_agent_runner
# agent = create_agent_runner()
# # inspect agent builder/tools for run_pipeline presence
# print("Has tracer:", hasattr(agent, "_tracer"))
# # print partial representation of where tools might be stored
# for name in ("tools", "builder", "executor", "agent_executor"):
#     obj = getattr(agent, name, None)
#     if obj is not None:
#         print(name, "->", obj)






# from app.services.agent_runner import create_agent_runner
# agent = create_agent_runner()
# print("Has tracer:", hasattr(agent, "_tracer"))
# print("Registered tools:", getattr(agent, "registered_tools", None))

# # Inspect builder and nodes for tool nodes (LangGraph-style)
# if hasattr(agent, "builder"):
#     b = agent.builder
#     print("builder type:", type(b))
#     # try common builder attributes that may contain tool info
#     for attr in ("tools", "_tools", "toolkit", "tools_map", "nodes"):
#         val = getattr(b, attr, None)
#         if val is not None:
#             print(f"builder.{attr} ->", type(val))
#             try:
#                 # attempt to list names
#                 if isinstance(val, (list, tuple)):
#                     print("  items:", [getattr(x, "name", getattr(x, "__name__", str(x))) for x in val][:20])
#                 elif isinstance(val, dict):
#                     print("  keys:", list(val.keys())[:20])
#             except Exception:
#                 pass

# # Also list top-level attrs for manual inspection
# print("agent keys:", list(vars(agent).keys())[:80])