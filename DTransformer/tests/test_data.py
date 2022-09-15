from ..data import KTData


def test_data():
    data = KTData("data/anonymized_full_release_competition_dataset/train.txt", 5)
    iterator = iter(data)
    (
        q,
        _,
        s,
        _,
    ) = next(iterator)
    (
        _,
        at,
        _,
        it,
    ) = next(iterator)

    assert q.size() == s.size()
    assert at.size() == it.size()
