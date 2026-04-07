"""Tests for HF upload functions with mocked API calls."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from ocelgen.upload.hf_upload import (
    build_repo_id,
    create_or_update_collection,
    upload_to_hub,
    upload_unified_dataset,
)


class TestBuildRepoId:
    def test_builds_correct_id(self) -> None:
        assert build_repo_id("myuser") == "myuser/open-agent-traces"


class TestUploadUnifiedDataset:
    @patch("ocelgen.upload.hf_upload.HfApi")
    def test_returns_url(self, mock_api_cls: MagicMock, tmp_path: Path) -> None:
        mock_api = mock_api_cls.return_value
        mock_api.create_repo.return_value = None
        mock_api.upload_folder.return_value = None

        url = upload_unified_dataset(tmp_path, "testns")

        assert url == "https://huggingface.co/datasets/testns/open-agent-traces"
        mock_api.create_repo.assert_called_once_with(
            "testns/open-agent-traces",
            repo_type="dataset",
            exist_ok=True,
        )
        mock_api.upload_folder.assert_called_once()


class TestUploadToHub:
    @patch("ocelgen.upload.hf_upload.HfApi")
    def test_returns_url(self, mock_api_cls: MagicMock, tmp_path: Path) -> None:
        mock_api = mock_api_cls.return_value
        mock_api.create_repo.return_value = None
        mock_api.upload_folder.return_value = None

        url = upload_to_hub(tmp_path, "testns", "my-domain")

        assert url == "https://huggingface.co/datasets/testns/agent-traces-my-domain"
        mock_api.create_repo.assert_called_once()


class TestCreateOrUpdateCollection:
    @patch("ocelgen.upload.hf_upload.HfApi")
    def test_creates_new_collection(self, mock_api_cls: MagicMock) -> None:
        mock_api = mock_api_cls.return_value
        mock_api.list_collections.return_value = []
        mock_collection = MagicMock()
        mock_collection.slug = "testns/open-agent-traces-abc123"
        mock_api.create_collection.return_value = mock_collection
        mock_api.add_collection_item.return_value = None

        url = create_or_update_collection("testns", "open-agent-traces", ["testns/repo1"])

        assert "collections" in url
        mock_api.create_collection.assert_called_once()
        mock_api.add_collection_item.assert_called_once()

    @patch("ocelgen.upload.hf_upload.HfApi")
    def test_updates_existing_collection(self, mock_api_cls: MagicMock) -> None:
        mock_api = mock_api_cls.return_value
        existing = MagicMock()
        existing.slug = "testns/open-agent-traces-abc123"
        mock_api.list_collections.return_value = [existing]
        mock_api.add_collection_item.return_value = None

        url = create_or_update_collection("testns", "open-agent-traces", ["testns/repo1"])

        assert "collections" in url
        mock_api.create_collection.assert_not_called()

    @patch("ocelgen.upload.hf_upload.HfApi")
    def test_handles_list_collections_error(self, mock_api_cls: MagicMock) -> None:
        mock_api = mock_api_cls.return_value
        mock_api.list_collections.side_effect = Exception("API error")
        mock_collection = MagicMock()
        mock_collection.slug = "testns/open-agent-traces-abc123"
        mock_api.create_collection.return_value = mock_collection

        url = create_or_update_collection("testns", "open-agent-traces", [])

        assert "collections" in url
        mock_api.create_collection.assert_called_once()

    @patch("ocelgen.upload.hf_upload.HfApi")
    def test_handles_add_item_error(self, mock_api_cls: MagicMock) -> None:
        mock_api = mock_api_cls.return_value
        mock_api.list_collections.return_value = []
        mock_collection = MagicMock()
        mock_collection.slug = "testns/open-agent-traces-abc123"
        mock_api.create_collection.return_value = mock_collection
        mock_api.add_collection_item.side_effect = Exception("item error")

        # Should not raise
        url = create_or_update_collection("testns", "open-agent-traces", ["testns/repo1"])
        assert "collections" in url
