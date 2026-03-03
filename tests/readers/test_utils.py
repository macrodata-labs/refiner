from refiner.sources.readers.utils import align_byte_range_to_newlines


def test_align_byte_range_to_newlines_includes_line_starts(tmp_path):
    # Lines:
    # 0:a\n
    # 2:bb\n
    # 5:ccc\n
    data = b"a\nbb\nccc\n"
    p = tmp_path / "x.txt"
    p.write_bytes(data)

    with p.open("rb") as fh:
        # Planned range that starts at 0 should include first line start.
        assert align_byte_range_to_newlines(fh, start=0, end=2, size=len(data)) == (
            0,
            2,
        )

        # Planned range inside first line should be empty (no line start in [1,2)).
        assert align_byte_range_to_newlines(fh, start=1, end=2, size=len(data)) is None

        # Planned range that covers start of "bb" should include that line.
        assert align_byte_range_to_newlines(fh, start=2, end=5, size=len(data)) == (
            2,
            5,
        )
