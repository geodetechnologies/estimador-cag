# Synthetic stress PDFs

Generate a deterministic, text-extractable attachment in an **existing** output
directory, using the project's Python environment:

```sh
.venv/bin/python -m evals.stress.fixtures.build_pdfs --size 5 --output-dir /path/to/output
```

`--size` accepts 0, 5, 20, 50 or 100. Size 0 represents **no attachment**:
it creates no file or directory. Other sizes create `stress-<size>kb.pdf`;
the command refuses to overwrite an existing file. Select a disposable output
directory and do not commit generated PDFs. Run with `--help` for options.

| Size label | PDF bytes (KiB = 1,024 bytes) | Pages |
| --- | ---: | ---: |
| 0 KB | 0 (no attachment) | 0 |
| 5 KB | 5,120 | 2 |
| 20 KB | 20,480 | 8 |
| 50 KB | 51,200 | 20 |
| 100 KB | 102,400 | 20 |

The command reports bytes and extracted characters separately. Compressed,
repeated synthetic text and PDF comment padding calibrate file bytes; file size
is not a proxy for extracted-text length or realistic document complexity.
The 100 KB case has roughly 60,000 extracted characters, not a demonstrated
truncation limit. The application has a 10 MiB attachment **byte** limit; this
generator does not exercise the application, a provider, or truncation.
