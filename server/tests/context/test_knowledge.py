from pathlib import Path

from app.context.knowledge import load_knowledge


def test_loads_docs_with_titles(tmp_path: Path):
    (tmp_path / "about.py").write_text('TITLE = "About Me"\nCONTENT = "Hello."\n')
    (tmp_path / "cv.py").write_text('CONTENT = "no title here"\n')
    (tmp_path / "__init__.py").write_text("")  # underscore files are skipped

    docs = load_knowledge(tmp_path)

    assert [d.slug for d in docs] == ["about", "cv"]
    assert docs[0].title == "About Me"
    assert docs[0].text == "Hello."
    assert docs[1].title == "cv"  # falls back to the filename
