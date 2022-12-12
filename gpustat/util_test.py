import sys
import pytest

from gpustat import util


def test_safecall():

    def _success():
        return 42

    def _error():
        raise FileNotFoundError("oops")

    assert util.safecall(_success, error_value=None) == 42
    assert util.safecall(_error, error_value=-1) == -1

    with pytest.raises(FileNotFoundError):
        # not catched because exc_types does not match
        assert util.safecall(_error, exc_types=ValueError, error_value=-1)

    assert util.safecall(_error, error_value=-1,
                         exc_types=FileNotFoundError) == -1
    assert util.safecall(_error, error_value=-1,
                         exc_types=(FileNotFoundError, OSError)) == -1


if __name__ == '__main__':
    sys.exit(pytest.main(["-s", "-v"] + sys.argv))
