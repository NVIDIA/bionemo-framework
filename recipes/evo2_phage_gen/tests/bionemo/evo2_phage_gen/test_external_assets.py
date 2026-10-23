# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-Apache2
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import hashlib
import io
import json
import sys
import tarfile
import time
import types
import zipfile
from pathlib import Path
from typing import ClassVar

import pytest

from bionemo.evo2_phage_gen import external_assets as assets
from bionemo.evo2_phage_gen.external_assets import (
    PreparedAsset,
    prepare_external_assets,
    prepare_phrogs_lookup,
    prepare_pyrodigal_wrapper,
)


class _Response(io.BytesIO):
    status = 200
    headers: ClassVar[dict[str, str]] = {"X-Release": "current"}

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()


def test_download_checks_provider_checksum_at_download_boundary(tmp_path: Path, monkeypatch) -> None:
    payload = b"provider archive"
    expected = hashlib.md5(payload, usedforsecurity=False).hexdigest()
    monkeypatch.setattr(assets.urllib.request, "urlopen", lambda _request, timeout=None: _Response(payload))

    output, headers = assets._download(
        "https://example.test/archive.tar.gz",
        tmp_path / "archive.tar.gz",
        published_md5=expected,
    )
    assert output.read_bytes() == payload
    assert headers["x-release"] == "current"

    monkeypatch.setattr(assets.urllib.request, "urlopen", lambda _request, timeout=None: _Response(b"changed"))
    with pytest.raises(ValueError, match="Published checksum"):
        assets._download(
            "https://example.test/archive.tar.gz",
            tmp_path / "changed.tar.gz",
            published_md5=expected,
        )


def test_download_sha256_checks_cache_and_transfer(tmp_path, monkeypatch):
    payload = b"pinned proteins"
    expected = hashlib.sha256(payload).hexdigest()
    monkeypatch.setattr(assets.urllib.request, "urlopen", lambda *_a, **_kw: _Response(payload))
    path = tmp_path / "proteins.tar.gz"
    assets._download("https://example.test/proteins", path, expected_sha256=expected)
    assert path.read_bytes() == payload
    path.write_bytes(b"wrong archive")
    with pytest.raises(ValueError, match="checksum"):
        assets._download("https://example.test/proteins", path, expected_sha256=expected)
    with pytest.raises(ValueError, match="checksum"):
        assets._download("https://example.test/proteins", tmp_path / "other", expected_sha256="0" * 64)
    assert not (tmp_path / "other").exists()


def test_member_db_build_and_reuse(tmp_path, monkeypatch):
    archive = tmp_path / "proteins.tar.gz"
    with tarfile.open(archive, "w:gz") as handle:
        payload = b">phrog_1 ## accession_a\nMAAA\n>phrog_1 ## accession_b\nMCCC\n"
        entry = tarfile.TarInfo("FAA_phrog/phrog_1.faa")
        entry.size = len(payload)
        handle.addfile(entry, io.BytesIO(payload))
    expected = hashlib.sha256(archive.read_bytes()).hexdigest()
    builds = []

    def fake_run(command, check):
        assert check
        assert command[1] == "createdb"
        fasta = Path(command[2]).read_text()
        # Individual proteins must stay distinct, not collapse to a consensus.
        identifiers = [line.split()[0] for line in fasta.splitlines() if line.startswith(">")]
        assert len(set(identifiers)) == 2
        assert "MAAA" in fasta and "MCCC" in fasta
        builds.append(fasta)
        output = Path(command[3])
        for suffix in ("", ".index", ".lookup", "_h", "_h.index", "_h.dbtype"):
            Path(f"{output}{suffix}").write_text("db")
        Path(f"{output}.dbtype").write_bytes((0).to_bytes(4, "little"))

    monkeypatch.setattr(assets.subprocess, "run", fake_run)
    options = dict(database_url=archive.as_uri(), expected_sha256=expected)
    first = assets.prepare_phrogs_member_db(tmp_path / "external", **options)
    second = assets.prepare_phrogs_member_db(tmp_path / "external", **options)
    assert first.path == second.path
    assert len(builds) == 1
    Path(f"{first.path}.index").unlink()
    assets.prepare_phrogs_member_db(tmp_path / "external", **options)
    assert len(builds) == 2  # A partial database is rebuilt, not accepted as ready.


def test_download_retries_stalled_transfer_and_resumes_partial(tmp_path: Path, monkeypatch, capsys) -> None:
    output_path = tmp_path / "archive.tar.gz"
    partial = tmp_path / "archive.tar.gz.part"
    partial.write_bytes(b"prefix")
    resumed = _Response(b"suffix")
    resumed.status = 206
    outcomes = iter((TimeoutError("stalled"), resumed))
    calls: list[tuple[str | None, float | None]] = []

    def fake_urlopen(request, timeout=None):
        calls.append((request.get_header("Range"), timeout))
        outcome = next(outcomes)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    monkeypatch.setattr(assets.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(time, "sleep", lambda _seconds: None)

    output, _headers = assets._download("https://example.test/archive.tar.gz", output_path)

    assert output.read_bytes() == b"prefixsuffix"
    assert calls == [("bytes=6-", 60.0), ("bytes=6-", 60.0)]
    messages = capsys.readouterr()
    assert "retrying" in messages.err
    assert "archive.tar.gz" in messages.out


def test_download_retries_clean_eof_from_content_length(tmp_path: Path, monkeypatch) -> None:
    """A short response that closes cleanly must not become a cached final archive."""
    truncated = _Response(b"abc")
    truncated.headers = {"Content-Length": "6"}
    complete = _Response(b"abcdef")
    complete.headers = {"Content-Length": "6"}
    outcomes = iter((truncated, complete))
    calls = 0

    def fake_urlopen(_request, timeout=None):
        nonlocal calls
        calls += 1
        return next(outcomes)

    monkeypatch.setattr(assets.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(time, "sleep", lambda _seconds: None)

    output, _headers = assets._download("https://example.test/archive.tar.gz", tmp_path / "archive.tar.gz")

    assert output.read_bytes() == b"abcdef"
    assert calls == 2


def test_pyrodigal_wrapper_is_a_normal_executable(tmp_path: Path) -> None:
    prepared = prepare_pyrodigal_wrapper(tmp_path / "bin")
    assert prepared.path.stat().st_mode & 0o111
    assert "exec pyrodigal" in prepared.path.read_text()


def test_native_tool_archives(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(assets, "_architecture", lambda: "x86_64")
    assert assets._select_tool_archive("mmseqs2-gpu", None) == (assets.DEFAULT_MMSEQS_GPU_URL, None)
    assert assets._select_tool_archive("diamond", None) == (assets.DEFAULT_DIAMOND_URL, None)
    assert assets._select_tool_archive("hmmer", None) == (assets.DEFAULT_HMMER_URL, None)
    assert assets._select_tool_archive("amrfinder", None) == (assets.DEFAULT_AMRFINDER_URL, None)
    assert assets._tool_extract_dir(tmp_path, "diamond").name == "diamond"

    monkeypatch.setattr(assets, "_architecture", lambda: "aarch64")
    assert assets._select_tool_archive("mmseqs2-gpu", None) == (
        "https://mmseqs.com/latest/mmseqs-linux-gpu-arm64.tar.gz",
        None,
    )
    assert assets._select_tool_archive("diamond", None) == (
        "https://conda.anaconda.org/bioconda/linux-aarch64/diamond-2.1.24-heed1f17_0.conda",
        "7c095bb36b6f99c494b7ae90757df423",
    )
    assert assets._select_tool_archive("hmmer", None) == (
        "https://conda.anaconda.org/bioconda/linux-aarch64/hmmer-3.4-hfe13ca0_4.tar.bz2",
        "6f0a04231ef9418215c516bd56ae6e68",
    )
    assert assets._select_tool_archive("amrfinder", None) == (
        "https://conda.anaconda.org/bioconda/linux-aarch64/ncbi-amrfinderplus-4.2.7-h9686939_0.conda",
        "592621df3f59ce1b562b962a6190496e",
    )
    assert assets._tool_extract_dir(tmp_path, "diamond").name == "diamond-aarch64"


def test_tool_archive_override(monkeypatch) -> None:
    monkeypatch.setattr(assets, "_architecture", lambda: "aarch64")

    assert assets._select_tool_archive("diamond", "https://example.test/custom.tar.gz") == (
        "https://example.test/custom.tar.gz",
        None,
    )

    monkeypatch.setattr(assets, "_architecture", lambda: "riscv64")
    with pytest.raises(RuntimeError, match="No diamond download is configured for riscv64"):
        assets._select_tool_archive("diamond", None)


def test_conda_payload_extraction(tmp_path: Path, monkeypatch) -> None:
    tar_buffer = io.BytesIO()
    with tarfile.open(fileobj=tar_buffer, mode="w") as archive:
        payload = b"native executable"
        member = tarfile.TarInfo("bin/tool")
        member.size = len(payload)
        archive.addfile(member, io.BytesIO(payload))

    package = tmp_path / "tool.conda"
    with zipfile.ZipFile(package, mode="w") as archive:
        archive.writestr("pkg-tool-0.tar.zst", tar_buffer.getvalue())

    class _IdentityDecompressor:
        @staticmethod
        def stream_reader(source):
            return source

    monkeypatch.setitem(
        sys.modules,
        "zstandard",
        types.SimpleNamespace(ZstdDecompressor=_IdentityDecompressor),
    )

    extracted = assets._extract_archive(package, tmp_path / "extracted")

    assert (extracted / "bin" / "tool").read_bytes() == b"native executable"


def test_derives_phrogs_consensus_from_pharokka_profiles(tmp_path: Path, monkeypatch) -> None:
    profile = tmp_path / "pharokka" / "phrogs_profile_db"
    profile.parent.mkdir()
    profile.write_text("profiles")
    Path(f"{profile}.lookup").write_text("0\tphrog_1\t0\n")
    for suffix in (".dbtype", ".index", "_h", "_h.dbtype", "_h.index"):
        Path(f"{profile}{suffix}").write_text("database")
    (profile.parent / "phrog_annot_v4.tsv").write_text("phrog\tannot\tcategory\n")
    (profile.parent / "VERSION_1_11_0").write_text("1.11.0\n")
    assert assets._find_pharokka_database(profile.parent) == profile.parent
    calls: list[list[str]] = []

    def fake_run(command, check):
        assert check
        calls.append(command)
        output = Path(command[3])
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("database")
        Path(f"{output}.dbtype").write_text("type")
        Path(f"{output}.lookup").write_text("0\tphrog_1\t0\n")

    monkeypatch.setattr(assets.subprocess, "run", fake_run)

    prepared = assets.prepare_phrogs_consensus_db(
        tmp_path / "external",
        bin_dir=tmp_path / "bin",
        profile_database=profile,
    )

    assert prepared.path.name == "phrogs_consensus_db_pad"
    assert [command[1] for command in calls] == ["profile2consensus", "makepaddedseqdb"]
    assert calls[1][-2:] == ["--write-lookup", "1"]


def test_phrogs_lookup_uses_available_profiles_and_tolerates_database_growth(tmp_path: Path) -> None:
    database = tmp_path / "phrogs_profile_db"
    database.write_text("db")
    Path(f"{database}.dbtype").write_text("type")
    Path(f"{database}.lookup").write_text("0\tphrog_1\t0\n1\tphrog_99\t0\n")
    annotations = tmp_path / "phrogs.tsv"
    annotations.write_text(
        "phrog\tannot\tcategory\n"
        "1\tintegrase\tintegration and excision\n"
        "2\texcisionase added by a newer release\tintegration and excision\n"
        "99\tcapsid\thead and packaging\n"
    )

    prepared = prepare_phrogs_lookup(annotations, database, tmp_path / "lysogeny.tsv")
    text = prepared.path.read_text()
    assert "phrog_1" in text
    assert "phrog_2" not in text
    assert "phrog_99" not in text


def test_phrogs_lookup_records_review_and_high_confidence_families(tmp_path: Path) -> None:
    database = tmp_path / "phrogs_profile_db"
    database.write_text("db")
    Path(f"{database}.dbtype").write_text("type")
    Path(f"{database}.lookup").write_text("0\tphrog_1\t0\n1\tphrog_2\t0\n")
    annotations = tmp_path / "phrogs.tsv"
    annotations.write_text(
        "phrog\tannot\tcategory\n"
        "1\tintegrase\tintegration and excision\n"
        "2\tunknown protein\tintegration and excision\n"
    )

    output = prepare_phrogs_lookup(annotations, database, tmp_path / "lookup.tsv").path.read_text()
    assert "high_confidence" in output
    assert "review" in output


def test_phrogs_safety_db_uses_current_lookup(tmp_path: Path) -> None:
    source = tmp_path / "phrogs_profile_db"
    source.write_text("full database")
    Path(f"{source}.lookup").write_text("7\tphrog_1\t0\n9\tphrog_99\t0\n")
    for suffix in (".dbtype", ".index", ".source", "_h", "_h.dbtype", "_h.index"):
        Path(f"{source}{suffix}").write_text("database")
    lookup = tmp_path / "safety.tsv"
    lookup.write_text(
        "phrog\tannot\tcategory\tconfidence\tmatched_term\n"
        "phrog_1\tintegrase\tintegration and excision\thigh_confidence\tintegrase\n"
    )
    selected_keys: list[str] = []

    def fake_run(command, **_kwargs):
        selected_keys.append(Path(command[2]).read_text())
        output = Path(command[4])
        output.write_text("selected database")
        Path(f"{output}.dbtype").write_text("type")
        Path(f"{output}.index").write_text("index")
        for suffix in (".lookup", ".source", "_h", "_h.dbtype", "_h.index"):
            Path(f"{output}{suffix}").symlink_to(Path(f"{source}{suffix}"))

    prepared = assets.prepare_phrogs_safety_db(
        source,
        lookup,
        tmp_path / "safety" / "phrogs_profile_db",
        mmseqs_path=tmp_path / "bin" / "mmseqs",
        runner=fake_run,
    )

    assert selected_keys == ["7\n"]
    assert prepared.path.read_text() == "selected database"
    assert prepared.detail == "1 selected family"


def test_minimal_asset_preparation_can_skip_network_work(tmp_path: Path) -> None:
    prepared = prepare_external_assets(
        tmp_path,
        download_mmseqs=False,
        download_dustmasker=False,
        download_diamond=False,
        download_hmmer=False,
        download_phrogs_annotation=False,
        download_arc_evo2=False,
        download_large_databases=False,
        download_checkv=False,
        configure_lovis4u=False,
    )
    assert [item.name for item in prepared] == ["prodigal"]


def test_safety_state_records_current_versions_and_paths(tmp_path: Path, monkeypatch) -> None:
    external = tmp_path / "external"
    bin_dir = external / "bin"
    amr_tool_dir = external / "tools" / "amrfinder_v4.2.7"
    amr_dir = external / "safety" / "amrfinder"
    toxin_dir = external / "safety" / "toxins"
    phrogs_dir = external / "phrogs"
    for directory in (bin_dir, amr_tool_dir, amr_dir, toxin_dir, phrogs_dir):
        directory.mkdir(parents=True, exist_ok=True)
    (amr_tool_dir / "amrfinder").write_text("amrfinder")
    (bin_dir / "amrfinder").symlink_to(amr_tool_dir / "amrfinder")
    for tool in ("diamond", "mmseqs"):
        (bin_dir / tool).write_text(tool)
    (amr_dir / "state.json").write_text(
        json.dumps(
            {
                "tool_version": "4.2.7",
                "database_path": str(amr_dir / "2026-08"),
                "database_version": "2026-08",
                "release_url": "https://example.test/amr",
            }
        )
    )
    (toxin_dir / "state.json").write_text(
        json.dumps(
            {
                "release": "UniProt current",
                "diamond_database_path": str(toxin_dir / "toxins.dmnd"),
            }
        )
    )
    profile = phrogs_dir / "profiles"
    lookup = phrogs_dir / "lookup.tsv"
    profile.write_text("db")
    lookup.write_text("lookup")
    monkeypatch.setattr(assets, "_tool_version", lambda path, *args: f"{path.name} current")

    state = assets._safety_state(
        external,
        bin_dir,
        PreparedAsset("profile", profile, "PHROGs current"),
        PreparedAsset("lookup", lookup, "one family"),
    )
    assert state["tools"]["amrfinder"]["path"] == str((bin_dir / "amrfinder").absolute())
    assert state["tools"]["diamond"]["version"] == "diamond current"
    assert state["databases"]["phrogs"]["release"] == "PHROGs current"


def test_parser_allows_current_database_overrides() -> None:
    parsed = assets.build_parser().parse_args(
        [
            "--with-safety",
            "--pharokka-database-url",
            "https://example.test/pharokka-databases.tar.gz",
            "--pharokka-database-md5",
            "provider-value",
            "--pharokka-database-release",
            "Pharokka newer",
            "--amrfinder-url",
            "https://example.test/amr.tar.gz",
            "--amrfinder-release",
            "amrfinder newer",
        ]
    )
    assert parsed.pharokka_database_release == "Pharokka newer"
    assert parsed.amrfinder_release == "amrfinder newer"
