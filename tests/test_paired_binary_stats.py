import pytest

from embodied_data_lab.paired_binary_stats import paired_risk_difference


@pytest.mark.parametrize(
    ("cells", "expected"),
    [
        ((36, 12, 2, 0), (0.0569, 0.3404)),
        ((20, 12, 2, 16), (0.0562, 0.3292)),
        ((36, 14, 0, 0), (0.1528, 0.4167)),
        ((54, 0, 0, 0), (-0.0664, 0.0664)),
        ((30, 0, 0, 24), (-0.0358, 0.0358)),
    ],
)
def test_newcombe_method_10_matches_published_reference_table(cells, expected):
    result = paired_risk_difference(*cells)
    assert result["confidence_interval"] == pytest.approx(expected, abs=5e-5)


def test_exact_mcnemar_uses_only_discordant_pairs():
    result = paired_risk_difference(30, 10, 0, 10)
    assert result["risk_difference"] == pytest.approx(0.2)
    assert result["discordant_pairs"] == 10
    assert result["mcnemar_exact_two_sided_p"] == pytest.approx(2 / 1024)


def test_paired_table_rejects_invalid_counts():
    with pytest.raises(ValueError):
        paired_risk_difference(0, 0, 0, 0)
    with pytest.raises(ValueError):
        paired_risk_difference(1, -1, 0, 0)
