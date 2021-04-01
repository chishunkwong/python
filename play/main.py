from solution import Solution
from ebird import db_loader


def main_not():
    db_loader.insert('US', 'NC')
    # db_loader.create_tables()


def main():
    sol = Solution()
    # print(sol.increasingTriplet([2, 1, 5, 0, 4, 6]))
    # print(sol.increasingTriplet([1, 5, 0, 4, 1, 3]))
    # print(sol.increasingTriplet([4, 5, 3, 4, 1, 2]))
    # arr = [1, 1, 1, 4, 5, 6, 7]
    # arr = [1, 1, 1, 4, 5, 6]
    # print(Solution.max_profit([3, 2, 3, 1, 4, 2, 6]))
    # print(Solution.max_profit([9, 7, 5, 4, 1]))
    # print(sol.maxSlidingWindow([1, 3, -1, -3, 5, 3, 6, 7], 3))
    # print(sol.maxSlidingWindow([9, 8, 7, 6, 5, 4, 3, 2, 1], 3))
    print(sol.generateParenthesis(4))


def test() -> None:
    add = get_add()
    print(add(2, 3))


def get_int() -> int:
    return 1


def get_add():
    def add(a: int, b: int) -> int:
        return a + b

    return add


def ref2(a: int):
    print(a)
    a = a + 10


if __name__ == '__main__':
    main()
