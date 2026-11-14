"""Code block of this repository's outputs: its own checkout plus the RESUM_FLEX checkout."""

from core.provenance import SOURCE as RESUM_FLEX_SOURCE

from legend_optical_map_ml.provenance import GIT_FIELDS, PACKAGES, SOURCE, code_block


def test_code_block_describes_both_checkouts():
    code = code_block()
    assert code["source"] == str(SOURCE)
    assert set(code["packages"]) == set(PACKAGES) and code["packages"]["h5py"]
    assert code["resum_flex"]["source"] == str(RESUM_FLEX_SOURCE)
    assert set(code["resum_flex"]) <= set(GIT_FIELDS)
    # Both repositories are git checkouts here: commits are recorded, not errors.
    assert len(code["commit"]) == 40 and len(code["resum_flex"]["commit"]) == 40
