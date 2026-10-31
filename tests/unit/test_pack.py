from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from app.config import get_settings
from app.tools.common import PACKS_ROOT, pack_directory
from app.tools.fees import FeeSchedule
from app.tools.fx import FxRate
from app.tools.instruments import InstrumentRow
from app.tools.pack_schemas import DocumentMetadata, PolicyPack
from app.tools.search import _read_document

PACK_DIR = PACKS_ROOT / "eu_us_etf"


def test_pack_configs_and_tabular_data_validate_as_fictional() -> None:
    fees = FeeSchedule.model_validate(
        yaml.safe_load((PACK_DIR / "fees.yaml").read_text(encoding="utf-8"))
    )
    policy = PolicyPack.model_validate(
        yaml.safe_load((PACK_DIR / "policy.yaml").read_text(encoding="utf-8"))
    )
    with (PACK_DIR / "instruments.csv").open(encoding="utf-8") as file:
        import csv

        instruments = [
            InstrumentRow.model_validate(row) for row in csv.DictReader(file)
        ]
    with (PACK_DIR / "fx_rates.csv").open(encoding="utf-8") as file:
        import csv

        rates = [FxRate.model_validate(row) for row in csv.DictReader(file)]

    assert fees.fictional_data is True
    assert fees.currency == "EUR"
    assert policy.fictional_data is True
    assert "fictional" in policy.disclaimer.casefold()
    assert len(instruments) >= 1
    assert all(item.symbol.startswith("DEMO-") for item in instruments)
    assert len(rates) >= 1
    assert all(rate.rate_date == "2025-01-15" for rate in rates)


def test_every_knowledge_document_has_valid_front_matter_and_length() -> None:
    docs_dir = PACK_DIR / "docs"
    paths = sorted(docs_dir.glob("*.md"))

    assert 15 <= len(paths) <= 25
    identifiers: set[str] = set()
    for path in paths:
        front_matter, body = _read_document(path)
        metadata = DocumentMetadata.model_validate(yaml.safe_load(front_matter))
        word_count = len(body.split())

        assert metadata.doc_id not in identifiers
        assert metadata.doc_id == path.stem
        assert metadata.verified is False
        assert 150 <= word_count <= 400, f"{path.name} has {word_count} words"
        if not metadata.fictional_data:
            assert metadata.source_url == ""
            assert metadata.retrieved_on == ""
            assert "TODO(verify)" in body
        identifiers.add(metadata.doc_id)


def test_fictional_docs_are_labelled_and_injection_fixture_is_inert() -> None:
    injection_path = PACK_DIR / "docs" / "prompt-injection-example.md"
    front_matter, body = _read_document(injection_path)
    metadata = DocumentMetadata.model_validate(yaml.safe_load(front_matter))
    fees_doc_path = PACK_DIR / "docs" / "demobroker-fees.md"
    fees_front_matter, _ = _read_document(fees_doc_path)
    fees_metadata = DocumentMetadata.model_validate(yaml.safe_load(fees_front_matter))

    assert metadata.fictional_data is True
    assert "untrusted document content" in body
    assert "Ignore all previous instructions" in body
    assert fees_metadata.fictional_data is True
    assert "fictional" in (PACK_DIR / "fees.yaml").read_text(encoding="utf-8")


def test_knowledge_front_matter_schema_rejects_unverified_claims() -> None:
    with pytest.raises(ValidationError):
        DocumentMetadata.model_validate(
            {
                "doc_id": "bad-id",
                "title": "Bad metadata",
                "fictional_data": False,
                "source_url": "",
                "retrieved_on": "",
                "verified": True,
            }
        )

    with pytest.raises(ValidationError, match="provided together"):
        DocumentMetadata.model_validate(
            {
                "doc_id": "partial-source",
                "title": "Incomplete source metadata",
                "fictional_data": False,
                "source_url": "https://example.test/source",
                "retrieved_on": "",
                "verified": False,
            }
        )


def test_pack_paths_are_inside_repository(monkeypatch: pytest.MonkeyPatch) -> None:
    assert PACK_DIR.parent == PACKS_ROOT
    assert Path(PACKS_ROOT).is_absolute()
    monkeypatch.setenv("PACK", "../README")
    get_settings.cache_clear()
    try:
        with pytest.raises(ValueError, match="Invalid pack name"):
            pack_directory()
    finally:
        get_settings.cache_clear()
