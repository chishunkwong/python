from typing import List
from collections import deque


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

    @staticmethod
    def max_profit(prices: List[int]):
        p_len = len(prices)
        if p_len < 2:
            return 0
        # Python has no smallest integer, so this has to do
        best = prices[1] - prices[0]
        if p_len == 2:
            return best
        cur_low = prices[0]
        for i in range(1, p_len - 1):
            p = prices[i]
            cur_low = min(cur_low, p)
            best = max(best, prices[i + 1] - cur_low)
        return best

    def maxSlidingWindow_slow(self, nums: List[int], k: int) -> List[int]:
        """
        Given an array of integers, return the list of maximum of each sliding window of length k,
        starting from left to right. So if nums has length l then the result should be
        an array of length l - k + 1.
        In particular, if k == l then the result is just the max of the array,
        and if k == 1 the result is just the array itself
        """
        nums_len = len(nums)
        if k < 1 or k > nums_len:
            return None
        if k == 1:
            return nums

        def max_and_index(numbers, start, end):
            """
            Given an array of integers and a start and end (exclusive),
            return the maximum in that range and the index that the maximum is at.
            In case of ties, this function favors the one later in the array
            """
            ind = start
            cur_max = numbers[start]
            for i in range(start + 1, end):
                n = nums[i]
                if n >= cur_max:
                    cur_max = n
                    ind = i
            return cur_max, ind

        result = list()
        last_max = -9999999
        last_max_at = -1
        for i in range(0, nums_len - k + 1):
            window_rightmost_at = i + k - 1
            window_rightmost = nums[window_rightmost_at]
            if last_max_at == i - 1:
                # darn, the max from the last window is at the left most, have to recalculate
                last_max, last_max_at = max_and_index(nums, i, i + k)
            elif last_max <= window_rightmost:
                # the rightmost won (tie is won by the right), so advance
                last_max, last_max_at = window_rightmost, window_rightmost_at
            else:
                # the maximum stays (and its index)
                pass
            result.append(last_max)

        return result

    def maxSlidingWindow2(self, nums: List[int], k: int) -> List[int]:
        """
        See function above, this one should be much faster
        (no it is not, failed even earlier than the last one)
        """
        nums_len = len(nums)
        if k < 1 or k > nums_len:
            return None
        if k == 1:
            return nums
        if k == nums_len:
            return [max(nums)]
        with_index = [(nums[i], i) for i in range(0, nums_len)]
        with_index.sort(key=lambda t: t[0], reverse=True)
        # Fill with a number that is smaller than the constraint minimum
        result = [-999999] * (nums_len - k + 1)
        filled = 0
        for next_largest, i in with_index:
            # if the next largest is bigger than what's in the result, replace it
            sphere_of_influence = range(max(i - k + 1, 0), min(i + 1, nums_len - k + 1))
            for j in sphere_of_influence:
                if next_largest > result[j]:
                    result[j] = next_largest
                    filled = filled + 1
            if filled >= nums_len - k + 1:
                print("breaking early", filled, next_largest)
                break
        return result

    def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
        """
        See the two functions above, this one will work, because it is not my algo but rather
        from leetcode's discussion (and using the problem's hints). Just really practicing
        """
        nums_len = len(nums)
        if k < 1 or k > nums_len:
            return None
        if k == 1:
            return nums
        if k == nums_len:
            return [max(nums)]
        answer = list()
        # last k entries in decreasing order, elements are tuples like (n, i),
        # where n is the number and i is the index, so in that sense all we need is i, because n can be had from i
        deq = deque()
        # first add the first k numbers to the deque, but drop any number
        # that has a number to the right of it that is at least as large as it
        for i in range(0, k):
            n = nums[i]
            # there is no peek in deque
            while deq and n >= deq[-1][0]:
                deq.pop()
            # add both the value and the index, because we will need to know
            # if a max happens at the left edge or not
            deq.append((n, i))
        answer.append(deq[0][0])
        # now the rest, follow similar logic as when we add the first k
        # but this time we also check if a number is outside the current window, if so we pop it as well
        for i in range(k, nums_len):
            n = nums[i]
            while deq:
                right_most = deq[-1]
                if right_most[0] <= n or right_most[1] <= i - k:
                    deq.pop()
                else:
                    break
            # print(deq, n, i)
            deq.append((n, i))
            # now take from the left until we find one that is in the current window,
            # while popping the ones that have become out of range
            while deq:
                left_most = deq[0]
                if left_most[1] <= i - k:
                    deq.popleft()
                else:
                    answer.append(left_most[0])
                    break

        return answer

    def generateParenthesis(self, n: int) -> List[str]:
        return self.generateParenthesisWithMem(n, dict())

    def generateParenthesisWithMem(self, n: int, haves: dict) -> List[str]:
        if n in haves:
            return haves[n]
        if n == 0:
            return list()
        if n == 1:
            return ["()"]
        answer = set()
        n_minus_ones = self.generateParenthesisWithMem(n - 1, haves)
        for s in n_minus_ones:
            answer.add(f"(){s}")
            answer.add(f"({s})")
            answer.add(f"{s}()")
        # any combination for i and n-i can sit next to each other, the example that broke
        # my original simple recursive answer was "(())(())" for n=4. I was just doing the three lines
        # above to go from n-1 to n
        for i in range(2, n - 1):
            eyes = self.generateParenthesisWithMem(i, haves)
            n_minus_eyes = self.generateParenthesisWithMem(n - i, haves)
            for s1 in eyes:
                for s2 in n_minus_eyes:
                    answer.add(f"{s1}{s2}")
        return list(answer)

    def search(self, nums: List[int], target: int) -> int:
        """
        https://leetcode.com/problems/search-in-rotated-sorted-array/
        search for a target in a sorted array that has been wrapped around once, like
        [4, 5, 1, 2, 3]
        """
        # first find the break, the list is supposed to be increasing, so take the half way point and compare
        # with first and last (binary search)
        l = len(nums)
        start = 0
        end = l - 1
        have_wrap = False
        while True:
            mid = int((start + end) / 2)
            print(mid, start, end)
            if nums[start] > nums[mid]:
                have_wrap = True
                end = mid
            elif nums[mid] > nums[end]:
                have_wrap = True
                start = mid
            else:
                break
            if end - start <= 0:
                break
            if end - start == 1:
                mid = end if nums[start] > nums[end] else start
                break
        print(mid)
        mid = mid if have_wrap else 0
        # Now that we found the wrapping point, it may be easier to just unwrap and then do a binary search,
        # but it would be slightly slower, so let's do it the hard way
        start = mid
        end = (mid + l - 1) % l
        counter = 0
        found = -1
        while counter < 100:
            counter = counter + 1
            sub_len = (end - start + 1) % l
            sub_len = l if sub_len == 0 else sub_len
            mid = int(start + sub_len / 2) % l
            print(mid, start, end, sub_len)
            if nums[mid] < target:
                start = mid
            elif nums[mid] > target:
                end = mid
            else:
                found = mid
                break
            if sub_len <= 2:
                if nums[start] == target:
                    found = start
                if nums[end] == target:
                    found = end
                break
        return found
