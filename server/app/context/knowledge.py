import importlib.util
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class KnowledgeDoc:
    title: str
    slug: str
    text: str


def _load_module(path: Path):
    spec = importlib.util.spec_from_file_location(f"knowledge.{path.stem}", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load knowledge module {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_knowledge(knowledge_dir: Path) -> list[KnowledgeDoc]:
    if not knowledge_dir.is_dir():
        raise FileNotFoundError(f"Knowledge directory not found: {knowledge_dir}")
    docs: list[KnowledgeDoc] = []
    # sorted() -> deterministic order (matters later for prompt caching)
    for path in sorted(knowledge_dir.glob("*.py")):
        if path.stem.startswith("_"):
            continue
        module = _load_module(path)
        title = getattr(module, "TITLE", path.stem)
        docs.append(KnowledgeDoc(title=title, slug=path.stem, text=module.CONTENT))
    return docs
