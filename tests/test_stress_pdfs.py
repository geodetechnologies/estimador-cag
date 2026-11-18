"""Exercise the PDF fixture generator through its command-line interface."""

import os
from pathlib import Path
import subprocess
import sys

import pytest
from pypdf import PdfReader


MODULE = 'evals.stress.fixtures.build_pdfs'
SENTINEL = 'SYNTHETIC STRESS FIXTURE'


def run_generator(size, output_dir):
    return subprocess.run(
        [sys.executable, '-m', MODULE, '--size', str(size), '--output-dir', str(output_dir)],
        capture_output=True, text=True, check=False,
        env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'},
    )


@pytest.mark.parametrize(('size', 'pages'), [(5, 2), (20, 8), (50, 20), (100, 20)])
def test_generates_deterministic_calibrated_text_pdfs(tmp_path, size, pages):
    first = tmp_path / 'first'
    second = tmp_path / 'second'
    first.mkdir()
    second.mkdir()
    results = [run_generator(size, directory) for directory in (first, second)]
    paths = [directory / f'stress-{size}kb.pdf' for directory in (first, second)]

    assert paths[0].read_bytes() == paths[1].read_bytes()
    assert len(paths[0].read_bytes()) == size * 1024
    reader = PdfReader(paths[0], strict=True)
    assert len(reader.pages) == pages
    texts = [page.extract_text() for page in reader.pages]
    assert all(SENTINEL in text for text in texts)
    characters = sum(map(len, texts))
    assert characters != size * 1024
    if size == 100:
        assert 50000 <= characters <= 60000
    for result, path in zip(results, paths):
        assert result.returncode == 0
        assert result.stderr == ''
        assert result.stdout == f'{size} KB: {path} ({size * 1024} bytes, {characters} extracted characters)\n'
        assert sorted(path.parent.iterdir()) == [path]


def test_zero_is_no_attachment_and_does_not_touch_output_directory(tmp_path):
    missing = tmp_path / 'missing'
    result = run_generator(0, missing)
    assert result.returncode == 0
    assert result.stdout == '0 KB: no attachment (0 bytes, 0 extracted characters)\n'
    assert result.stderr == ''
    assert not missing.exists()
    assert list(tmp_path.iterdir()) == []


def test_refuses_to_overwrite_existing_file(tmp_path):
    target = tmp_path / 'stress-5kb.pdf'
    target.write_bytes(b'user data')
    result = run_generator(5, tmp_path)
    assert result.returncode == 1
    assert result.stdout == ''
    assert result.stderr == f'Error: output already exists: {target}\n'
    assert target.read_bytes() == b'user data'
    assert sorted(tmp_path.iterdir()) == [target]


def test_rejects_unsupported_size_without_creating_files(tmp_path):
    result = run_generator(10, tmp_path)
    assert result.returncode == 2
    assert result.stdout == ''
    assert 'invalid choice: 10' in result.stderr
    assert list(tmp_path.iterdir()) == []


def test_help_explains_sizes_and_no_attachment():
    result = subprocess.run(
        [sys.executable, '-m', MODULE, '--help'], capture_output=True,
        text=True, check=False, env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'},
    )
    assert result.returncode == 0
    assert result.stderr == ''
    assert '--size' in result.stdout
    assert '--output-dir' in result.stdout
    assert '0 means no attachment' in result.stdout
    assert '5,20,50,100' in result.stdout
