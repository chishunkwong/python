from solution import Solution


def main():
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
