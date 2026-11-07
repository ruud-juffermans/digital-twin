from app.agent.tools import build_tools, tool_definitions
from app.context.knowledge import KnowledgeDoc


def make_doc(title: str, text: str) -> KnowledgeDoc:
    return KnowledgeDoc(title=title, slug=title.lower(), text=text)


def test_search_knowledge_returns_matching_snippets():
    docs = [
        make_doc("CV", "# CV\n\nRuud works with Kafka in Amsterdam.\n\nUnrelated paragraph."),
        make_doc("About", "# About\n\nHe sails on the IJsselmeer."),
    ]
    results = build_tools(docs)["search_knowledge"].run(query="kafka amsterdam")

    assert results[0] == {"document": "CV", "snippet": "Ruud works with Kafka in Amsterdam."}
    assert all("Unrelated" not in r["snippet"] for r in results)


def test_search_knowledge_returns_empty_for_no_match():
    tools = build_tools([make_doc("CV", "Nothing relevant here.")])
    assert tools["search_knowledge"].run(query="quantum blockchain") == []


def test_get_availability_returns_structured_data():
    value = build_tools([])["get_availability"].run()
    assert value["status"] == "open"
    assert value["available_from"] == "now"


def test_definitions_are_narrow_and_strict():
    definitions = tool_definitions(build_tools([]))

    assert {d["function"]["name"] for d in definitions} == {
        "search_knowledge", "get_availability", "draft_contact_message",
    }
    for definition in definitions:
        assert definition["type"] == "function"
        schema = definition["function"]["parameters"]
        assert schema["additionalProperties"] is False
        assert set(schema["required"]) == set(schema["properties"])


def test_only_the_contact_draft_is_high_risk():
    tools = build_tools([])
    assert [name for name, tool in tools.items() if tool.high_risk] == ["draft_contact_message"]
