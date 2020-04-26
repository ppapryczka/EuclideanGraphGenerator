from p1.file_in_package_1 import function_in_package_1
from p2.file_in_package_2 import function_in_package_2


def test_answer1():
    assert function_in_package_1(3) == 5


def test_answer2():
    assert function_in_package_2(3) == 5


def test_answer3():
    assert 2 + 2 == 4
