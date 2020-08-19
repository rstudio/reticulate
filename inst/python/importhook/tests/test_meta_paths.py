from importhook import HookMetaPaths


class DummyFinder:
    pass


def test_hook_meta_paths_setitem():
    finder1 = DummyFinder()
    finder2 = DummyFinder()
    paths = HookMetaPaths([finder1, finder2])
    finder3 = DummyFinder()
    paths[0] = finder3
    assert paths == [finder3, finder2]
    assert finder3.__hooked__ is True
