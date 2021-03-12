from typing import List


class Solution:
    def increasingTriplet(self, nums: List[int]) -> bool:
        # as we walk from left to right, track these numbers:
        # the smallest number in the array
        minimum: int = None
        # the smallest number in the array that has a number to the left that is smaller than it
        has_smaller: int = None
        for num in nums:
            if minimum is None or num < minimum:
                minimum = num
            if minimum is not None and num > minimum:
                # num is bigger than at least one number
                if has_smaller is None or has_smaller > num:
                    has_smaller = num
            if has_smaller is not None and num > has_smaller:
                return True
        return False
