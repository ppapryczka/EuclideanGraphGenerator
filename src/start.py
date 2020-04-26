from p1.file_in_package_1 import function_in_package_1
from p2.file_in_package_2 import function_in_package_2


def main():
    print("Program stated")
    print("aaa")
    print(function_in_package_1(2))
    print(function_in_package_2(2))


if __name__ == '__main__':
    main()
