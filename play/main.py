from solution import Solution
from ebird import db_loader


def main():
    db_loader.insert('US', 'WV')
    # db_loader.create_tables()


def main_not():
    sol = Solution()
    # print(sol.increasingTriplet([2, 1, 5, 0, 4, 6]))
    # print(sol.increasingTriplet([1, 5, 0, 4, 1, 3]))
    # print(sol.increasingTriplet([4, 5, 3, 4, 1, 2]))
    # arr = [1, 1, 1, 4, 5, 6, 7]
    # arr = [1, 1, 1, 4, 5, 6]
    arr = [4, 5, 5, 6]
    sol.wiggleSort(arr)
    print(arr)


if __name__ == '__main__':
    main()
