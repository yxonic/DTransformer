from ..data import KTData, KTDataWithPid


def test_data():
    data_path = "data/2009_skill_builder_data_corrected/train.txt"

    data = KTData(data_path)
    q, s = next(iter(data))
    assert q.size() == s.size()

    batch_size = 8
    data = KTData(data_path, batch_size=batch_size)
    q, s = next(iter(data))
    assert q.size(0) == batch_size
    assert q.size() == s.size()

    data_path = "data/anonymized_full_release_competition_dataset/train.txt"

    data = KTDataWithPid(data_path)
    q, s, pid, it, at = next(iter(data))
    assert q.size() == s.size()
    print(q.size())

    batch_size = 4
    data = KTDataWithPid(data_path, batch_size=batch_size, shuffle=True)
    q, s, pid, it, at = next(iter(data))
    assert q.size(0) == batch_size
    assert q.size() == s.size()
