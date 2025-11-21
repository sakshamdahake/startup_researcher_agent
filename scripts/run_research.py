from app.services.research_agent import ResearchAgent
agent = ResearchAgent(namespace="dev-tests", progress_cb=lambda t,p: print(t, p))
report = agent.run("subscription meal kits for busy parents", pdf_paths=[])
print("Final report title:", report.title)
print(report.model_dump_json(indent=2))