from solution import Solution
from ebird import db_loader


def main_not():
    db_loader.insert('US', 'NC')
    # db_loader.create_tables()


def main():
    sol = Solution()
    # print(sol.increasingTriplet([2, 1, 5, 0, 4, 6]))
    # print(Solution.max_profit([3, 2, 3, 1, 4, 2, 6]))
    # print(sol.maxSlidingWindow([1, 3, -1, -3, 5, 3, 6, 7], 3))
    # print(sol.generateParenthesis(4))
    # print(sol.search([4, 5, 6, 7, 0, 1, 2], 3))
    # print(sol.search([3], 4))
    print(sol.search([3, 1], 2))
    # print(sol.search([4, 5, 6, 7, 9, 10], 4))


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
