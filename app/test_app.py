import pytest


def test_streamlit_app_smoke():
    pytest.importorskip("streamlit")
    pytest.importorskip("plotly")
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file("streamlit_app.py", default_timeout=60).run()
    assert not at.exception
    assert at.title[0].value == "Эмбеддинги хостов: результаты"


def test_dashboard_renders_every_tab_for_the_demo_bundle(monkeypatch):
    pytest.importorskip("streamlit")
    pytest.importorskip("plotly")
    from pathlib import Path
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("PT_RESULTS_DIR", str(Path(__file__).resolve().parent / "testdata" / "synthetic_run"))
    at = AppTest.from_file("streamlit_app.py", default_timeout=120).run()
    assert not at.exception
    assert any("СИНТЕТИКА" in w.value and "не трафик LANL" in w.value for w in at.warning)   # пометка всегда есть
    assert len(at.tabs) == 7 and [t.label for t in at.tabs][:3] == ["Обзор", "Таблица", "Прогноз связей"]
    # взаимодействие: переключить группу метрик и окно эталона
    at.selectbox(key="lb_group").select("Будущие связи").run()
    assert not at.exception
    at.radio(key="lb_target").set_value("validation").run()
    assert not at.exception


def test_dashboard_with_missing_bundle_shows_a_clear_error(monkeypatch, tmp_path):
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file("streamlit_app.py", default_timeout=60)
    at.run()
    at.text_input[0].set_value(str(tmp_path)).run()
    assert not at.exception and any("run/manifest.json" in e.value for e in at.error)


def _app_with_bundle(monkeypatch, bundle):
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest
    monkeypatch.setenv("PT_RESULTS_DIR", str(bundle))
    return AppTest.from_file("streamlit_app.py", default_timeout=120).run()


def test_pt_results_dir_is_selected_by_default(monkeypatch, tmp_path):
    import shutil
    from pathlib import Path
    demo = Path(__file__).resolve().parent / "testdata" / "synthetic_run"
    shutil.copytree(demo, tmp_path / "mine")
    at = _app_with_bundle(monkeypatch, tmp_path / "mine")
    assert not at.exception and at.sidebar.selectbox[0].value == str(tmp_path / "mine")


def test_dashboard_degrades_gracefully_when_artifacts_are_missing(monkeypatch, tmp_path):
    import shutil
    from pathlib import Path
    demo = Path(__file__).resolve().parent / "testdata" / "synthetic_run"
    b = tmp_path / "b"
    shutil.copytree(demo, b)
    run = b / "run"
    (run / "audit.json").unlink()
    shutil.rmtree(run / "embeddings")
    shutil.rmtree(run / "identity")
    shutil.rmtree(run / "results" / "validation")
    (b / "reference" / "test" / "neighbors.csv").unlink()
    at = _app_with_bundle(monkeypatch, b)
    assert not at.exception
    text = " ".join(w.value for w in at.warning)
    assert "виды по хостам недоступны" in text and "нет audit.json" in text
    assert any("СИНТЕТИКА" in w.value for w in at.warning)          # пометка на месте


def test_unverified_and_declared_bundles_get_their_own_banner(monkeypatch, tmp_path):
    import json, shutil
    from pathlib import Path
    demo = Path(__file__).resolve().parent / "testdata" / "synthetic_run"
    for kind, expect_in, banner_kind in (("unverified", "НЕПРОВЕРЕННЫЕ ДАННЫЕ", "error"), ("lanl_declared", "заявлено оператором", "info")):
        b = tmp_path / kind
        shutil.copytree(demo, b)
        mf = b / "run" / "manifest.json"
        m = json.loads(mf.read_text())
        m.update(data_kind=kind, synthetic=False, real_lanl_declared=kind == "lanl_declared")
        m["provenance"] = {**(m.get("provenance") or {}), "kind": kind, "source_note": "unit test note"}
        mf.write_text(json.dumps(m))
        at = _app_with_bundle(monkeypatch, b)
        assert not at.exception
        texts = [e.value for e in getattr(at, banner_kind)]
        assert any(expect_in in t for t in texts), (kind, texts)
        assert not any("СИНТЕТИКА" in w.value for w in at.warning)
        assert not any("проверено" in t.lower() and "не проверено" not in t.lower() for t in texts)
