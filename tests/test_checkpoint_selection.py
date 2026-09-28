from tools.select_experiment1_checkpoint import select_checkpoint, selection_score


def candidate(epoch, absent_red, present_red, absent_blue=0.0, present_blue=0.0, passed=True):
    return {
        "epoch": epoch,
        "clean_d200_gate": {"passed": passed},
        "slices": {
            "marker_absent": {"red_rate": absent_red, "blue_rate": absent_blue},
            "marker_present": {"red_rate": present_red, "blue_rate": present_blue},
        },
    }


def test_selection_prefers_the_stronger_weak_slice():
    lopsided = candidate(100, 1.0, 0.70)
    balanced = candidate(200, 0.82, 0.82)
    assert selection_score(balanced) > selection_score(lopsided)
    assert select_checkpoint([lopsided, balanced]) is balanced


def test_selection_ties_resolve_by_total_red_then_blue_then_earlier_epoch():
    lower_total_red = candidate(100, 0.80, 0.70)
    higher_total_red = candidate(200, 0.90, 0.70)
    assert select_checkpoint([lower_total_red, higher_total_red]) is higher_total_red

    more_blue = candidate(100, 0.90, 0.70, 0.10, 0.10)
    less_blue = candidate(200, 0.90, 0.70, 0.02, 0.04)
    assert select_checkpoint([more_blue, less_blue]) is less_blue

    earlier = candidate(100, 0.90, 0.70, 0.02, 0.04)
    later = candidate(200, 0.90, 0.70, 0.02, 0.04)
    assert select_checkpoint([later, earlier]) is earlier


def test_selection_returns_none_when_no_checkpoint_passes():
    assert select_checkpoint([candidate(100, 0.9, 0.9, passed=False)]) is None
