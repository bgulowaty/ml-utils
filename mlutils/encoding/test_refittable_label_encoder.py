from mlutils.encoding.refittable_label_encoder import RefitableLabelEncoder


def test_adds_unseen_labels_on_fit():
    # given
    le = RefitableLabelEncoder()
    le.fit(['a', 'b'])
    encoded = le.transform(['a', 'b'])

    # when
    le.fit(['c', 'b'])
    actual = le.transform(['a', 'c', 'b'])

    # then
    assert (actual == [encoded[0], 2, encoded[1]]).all()

def test_adds_unseen_labels_on_fit_transform():
    # given
    le = RefitableLabelEncoder()
    encoded = le.fit_transform(['a', 'c'])

    # when
    actual = le.fit_transform(['a', 'c', 'b', 'zxc'])

    # then
    assert (actual == [encoded[0], encoded[1], 2, 3]).all()
