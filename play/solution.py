from typing import List


class Solution:

    def wiggleSort(self, nums: List[int]) -> None:
        """
        Given an integer array nums, reorder it such that nums[0] < nums[1] > nums[2] < nums[3]...
        Do not return anything, modify nums in-place instead.
        """
        # first we find the min and the max in the array
        arr_min = min(nums)
        arr_max = max(nums)
        max_keys = arr_max - arr_min + 1
        counts = [0] * max_keys
        for num in nums:
            counts[num - arr_min] = counts[num - arr_min] + 1
        l = len(nums)
        is_odd = len(nums) % 2 == 1
        half_ish = int(l / 2)
        half = (half_ish + 1) if is_odd else half_ish
        filled = 0
        # go through the counts one at a time starting from the low end, take half of them to fill the
        # even indices of the input array, then take the rest to fill the old indices.
        # we start from the back to make sure if there are ties around the median then they are well separated
        # E.g. [4, 5, 5, 6] -> [5, 6, 4, 5]
        for idx, val in enumerate(counts):
            for i in range(0, val):
                if filled < half:
                    # populating the even indices (0-based), the smaller half
                    if is_odd:
                        nums[l - 1 - filled * 2] = idx + arr_min
                    else:
                        nums[l - 2 - filled * 2] = idx + arr_min
                else:
                    # populating the odd indices (0-based), the larger half
                    if is_odd:
                        print(filled, half)
                        nums[l - 2 - (filled - half) * 2] = idx + arr_min
                    else:
                        nums[l - 1 - (filled - half) * 2] = idx + arr_min
                filled = filled + 1

    def increasingTriplet(self, nums: List[int]) -> bool:
        """
        Given an array, determine whether there are three elements that are strictly increasing
        in both indices and values (no need to be continuous indices)
        """
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
