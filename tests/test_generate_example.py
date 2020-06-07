from generate_example import generate_random_graph_command
import pytest
import os


def test_check_if_file_exists(tmpdir):
    """
    Check if function save image if correct arguments.
    """

    img_path = os.path.join(tmpdir, "aaa.png")
    args = ["-N", "10", "-R", "0.1", "-O", img_path]

    generate_random_graph_command(args)
    assert os.path.isfile(img_path)


def test_wrong_args():
    """
    Give function wrong argument.
    """

    args = ["-N", "10", "-L", "0.1"]

    with pytest.raises(SystemExit) as pytest_wrapped_e:
        generate_random_graph_command(args)
    assert pytest_wrapped_e.type == SystemExit
    assert pytest_wrapped_e.value.code == 2


def test_wrong_radius():
    """
    Give function too long radius.
    """

    args = ["-N", "10", "-R", "1.1"]

    with pytest.raises(SystemExit) as pytest_wrapped_e:
        generate_random_graph_command(args)
    assert pytest_wrapped_e.type == SystemExit
    assert pytest_wrapped_e.value.code == 2
