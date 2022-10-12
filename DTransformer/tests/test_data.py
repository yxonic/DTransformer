from ..data import KTData


def test_data():
    data_path = "data/assist09/train.txt"

    data = KTData(data_path, inputs=["pid", "q", "s"])
    q, s = data[2].get("q", "s")
    assert q.size() == s.size()

    batch_size = 8
    data = KTData(data_path, inputs=["pid", "q", "s"], batch_size=batch_size)
    q, s = next(iter(data)).get("q", "s")
    assert q.size(0) == batch_size
    assert q.size() == s.size()

    data_path = "data/assist17/train.txt"

    data = KTData(data_path, inputs=["q", "s", "pid", "it", "at"])
    q, s = next(iter(data)).get("q", "s")
    assert q.size() == s.size()

    batch_size = 4
    data = KTData(
        data_path,
        inputs=["q", "s", "pid", "it", "at"],
        batch_size=batch_size,
        shuffle=True,
    )
    q, s, at = next(iter(data)).get("q", "s", "at")
    assert q.size(0) == batch_size
    assert q.size() == s.size()
    assert q.size() == at.size()
