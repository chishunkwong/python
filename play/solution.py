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

    def rotate(self, matrix: List[List[int]]) -> None:
        """
        Rotate a square matrix clockwise 90 degrees, in place swapping, not returning anything
        """
        side = len(matrix)
        if side <= 1:
            return
        # no need to consider the extra 0.5 when side is odd, because in that case it will be a center
        # pixel (element) that does not rotate
        half = int(side / 2)
        for layer in range(0, half):
            for i in range(layer, side - layer - 1):
                # stash the one on the top
                swp = matrix[layer][i]
                # take the number on the left to the top
                matrix[layer][i] = matrix[side - i - 1][layer]
                # take the number on the bottom to the left
                matrix[side - i - 1][layer] = matrix[side - layer - 1][side - i - 1]
                # take the number on the right to the bottom
                matrix[side - layer - 1][side - i - 1] = matrix[i][side - layer - 1]
                # set the stashed top to the right
                matrix[i][side - layer - 1] = swp

    def searchRange(self, nums: List[int], target: int) -> List[int]:
        nums_len = len(nums)
        if not nums:
            return [-1, -1]
        if nums_len == 1:
            if nums[0] == target:
                return [0, 0]
            else:
                return [-1, -1]

        from typing import Tuple

        def bin_search(start: int, end: int) -> Tuple[int, int, int]:
            """
            find an index of target in nums within the given start and end (inclusive)
            :return: a tuple of 3 ints: the index, the start and the end in the last binary search
            that found the target
            """
            if end - start <= 1:
                if nums[start] == target:
                    return start, start, end
                elif nums[end] == target:
                    return end, start, end
                else:
                    return -1, start, end
            mid = int((start + end) / 2)
            at_mid = nums[mid]
            if at_mid > target:
                return bin_search(start, mid)
            elif at_mid < target:
                return bin_search(mid, end)
            else:
                return mid, start, end

        found, start, end = bin_search(0, nums_len - 1)
        if found == -1:
            return [-1, -1]
        found_first = found
        left_limit = found
        while found != -1 and found > start:
            found, start, _ = bin_search(start, found - 1)
            if found != -1:
                left_limit = found
        found = found_first
        right_limit = found
        while found != -1 and found < end:
            found, _, end = bin_search(found + 1, end)
            if found != -1:
                right_limit = found

        return [left_limit, right_limit]

    def jump(self, nums: List[int]) -> int:
        """
        https://leetcode.com/problems/jump-game-ii/
        Given an array of max allowed jumps at a particular location, find the minimum number of jumps to go
        from the 0-th index of the array to the last index of the array. E.g. [2,3,1,1,4] is 2 (from 0-th to 1-st
        i.e. use only 1 of the allowed 2, then jump 3 indices to reach the last index)
        """
        return Solution.min_jump_at_idx(nums, 0, dict())

    @staticmethod
    def min_jump_at_idx(nums: List[int], idx: int, haves: dict) -> int:
        """
        like the function above, but find the minimum jump starting from a particular index, also uses a
        dict to remember the minimum if we have already calculated it before
        """
        nums_len = len(nums)
        if idx == nums_len - 1:
            # we are there, so even if allowed jump is 0 we are still good
            return 0
        allowed = nums[idx]
        if allowed == 0:
            # we will never get there, so return just a very big number
            return 9999999999
        if idx + allowed >= nums_len:
            # print("reached:", idx, allowed)
            return 1
        if idx in haves:
            return haves[idx]
        candidates = list()
        for jump in range(1, allowed + 1):
            candidates.append(1 + Solution.min_jump_at_idx(nums, idx + jump, haves))
        answer = min(candidates)
        haves[idx] = answer
        # print("partial:", idx, answer)
        return answer

    def permute(self, nums: List[int]) -> List[List[int]]:
        """
        Given an array nums of distinct integers, return all the possible permutations, in any order.
        """
        seed = list()
        seed.append([])
        return Solution.permute_recur(nums, seed)

    @staticmethod
    def permute_recur(nums: List[int], answers) -> List[List[int]]:
        nums_len = len(nums)
        if nums_len == 1:
            only_num = nums[0]
            for a_list in answers:
                a_list.append(only_num)
            return answers
        else:
            combined_answers = list()
            for i in range(0, nums_len):
                answers_clone = list()
                for l in answers:
                    # deep clone
                    answers_clone.append(list(l))
                num = nums[i]
                for a_list in answers_clone:
                    a_list.append(num)
                ans_i = Solution.permute_recur(nums[0:i] + nums[i + 1: nums_len], answers_clone)
                combined_answers.extend(ans_i)
            return combined_answers

    def findMedianSortedArrays(self, nums1: List[int], nums2: List[int]) -> float:
        """
        Given two sorted arrays nums1 and nums2 of size m and n respectively,
        return the median of the two sorted arrays.
        At least one of the two arrays can be assumed to be non-empty
        """
        len1 = len(nums1)
        len2 = len(nums2)
        if len1 == 0:
            return Solution.simple_median(nums2)
        if len2 == 0:
            return Solution.simple_median(nums1)
        len_all = len1 + len2
        is_odd = len_all % 2 == 1
        half = int((len_all + 1) / 2 if is_odd else len_all / 2)
        if is_odd:
            median, _ = Solution.find_value_at_sorted_index(nums1, nums2, half)
        else:
            below, above = Solution.find_value_at_sorted_index(nums1, nums2, half, True)
            median = (below + above) / 2
        return median

    @staticmethod
    def find_value_at_sorted_index(nums1: List[int], nums2: List[int], idx: int, need_next: bool = False) -> float:
        """
        Given two sorted, non-empty, int arrays, and an int that is between 0 and len1 + len2, find the value
        at combined index of that int, if the two arrays were to be put together and sorted
        Note that idx is 1-based
        """
        len1 = len(nums1)
        len2 = len(nums2)
        start1 = 0
        start2 = 0
        found = None
        while found is None:
            # need_ahead is how much we need to advance, starting from the two respective starts, to reach
            # the number at idx (if the arrays were put together and sorted)
            need_ahead = idx - start1 - start2 - 2
            if need_ahead <= 0:
                at1 = nums1[start1]
                at2 = nums2[start2]
                if need_ahead == -1:
                    # this is the special case where both arrays have one element
                    found = min(at1, at2)
                    if need_next:
                        found = found, max(at1, at2)
                    else:
                        found = found, None
                    break
                # if we get here it means we have reached the decision point where we check to see
                # if the idx element should be the larger at-x or should be the one after the smaller at-x,
                # or neither, then we keep going
                one_more1 = None if start1 + 1 == len1 else nums1[start1 + 1]
                one_more2 = None if start2 + 1 == len2 else nums2[start2 + 1]
                one_less1 = None if start1 == 0 else nums1[start1 - 1]
                one_less2 = None if start2 == 0 else nums2[start2 - 1]
                if at1 == at2:
                    # this case is probably not needed because we should have checked this condition already
                    found = at1
                elif at1 < at2:
                    if one_more1 is None or one_more1 >= at2:
                        found = at2
                    elif start2 == 0 or one_more1 >= one_less2:
                        found = one_more1
                        start1 = start1 + 1
                        start2 = start2 - 1
                else:
                    # vice versa
                    if one_more2 is None or one_more2 >= at1:
                        found = at1
                    elif start1 == 0 or one_more2 >= one_less1:
                        found = one_more2
                        start1 = start1 - 1
                        start2 = start2 + 1
                if found is not None:
                    if need_next:
                        found = found, Solution.find_next_value(nums1, nums2, start1, start2)
                    else:
                        found = found, None
                    break
            # If we get here we are not close enough yet, so we advance within each array, by giving
            # half of need_ahead to each array, taking into account one of the arrays may overflow.
            # It may also be that we have reached start1 + start2 + 2 = idx, but the two sides are imbalanced,
            # if so we try again by taking from the high side and give to the low side
            if need_ahead > 0:
                half_need = int(need_ahead / 2)
                if len1 - start1 <= len2 - start2:
                    # first array has fewer numbers (counting from the current start), so we take
                    # from it first as it is the one that may not have enough elements to walk ahead of
                    take1 = half_need if start1 + half_need < len1 else len1 - start1 - 1
                    take2 = need_ahead - take1
                else:
                    # vice versa
                    take2 = half_need if start2 + half_need < len2 else len2 - start2 - 1
                    take1 = need_ahead - take2
                start1 = start1 + take1
                start2 = start2 + take2
            at1 = nums1[start1]
            at2 = nums2[start2]
            print(at1, at2, start1, start2, idx)
            if at1 == at2:
                # we got lucky, the two arrays have the same value at the two advanced indices,
                # so we know either one can be our value at idx.
                found = at1
                if need_next:
                    found = found, Solution.find_next_value(nums1, nums2, start1, start2)
                else:
                    found = found, None
            elif at1 > at2:
                # in case nums2 is actually exhausted, then at1 is already the answer
                # (+1 because start2 is 0-based, so say start2 = 0 then we need to add 1 to get len2 of 1)
                if start2 + 1 == len2:
                    found = at1
                    if need_next:
                        found = found, Solution.find_next_value(nums1, nums2, start1, start2)
                    else:
                        found = found, None
                else:
                    # we took two samples, one from each array, and found that the one at array1 is larger
                    # so we step back in array1. I.e. we change start1 to the index in array1 where the value is
                    # the first such that is larger than at2 (we know it must exist, because worst case the value
                    # will be at1). Now we try again by returning to the top of the loop
                    old_start1 = start1
                    start1 = Solution.find_target_index(nums1, at2 + 0.5)
                    start2 = start2 + (old_start1 - start1)
                    if start2 >= len2:
                        diff = start2 - len2 + 1
                        start2 = start2 - diff
                        start1 = start1 + diff
            else:
                # must be at2 > at1, because we already dealt with the equal case
                # do the same thing as above, just 1 is 2 and 2 is 1
                if start1 + 1 == len1:
                    found = at2
                    if need_next:
                        found = found, Solution.find_next_value(nums1, nums2, start1, start2)
                    else:
                        found = found, None
                else:
                    old_start2 = start2
                    start2 = Solution.find_target_index(nums2, at1 + 0.5)
                    start1 = start1 + (old_start2 - start2)
                    if start1 >= len1:
                        diff = start1 - len1 + 1
                        start1 = start1 - diff
                        start2 = start2 + diff

        return found

    @staticmethod
    def find_next_value(nums1: List[int], nums2: List[int], idx1: int, idx2: int) -> float:
        """
        Given two sorted integer arrays and one index for each array, find the next value in the combined,
        sorted, array. It is assumed that the two arrays when sliced to include up to the respective index, and
        then combined, will constitute a contiguous slice in the combined, sorted, array.
        The indices can be -1, which means that the corresponding array is not contributing any element to the
        aforementioned combination, and the next value from that array is the 0-th element.
        Likewise it can also be len-1, which means that array cannot possibly provide a next value (trivial case then)
        """
        len1 = len(nums1)
        len2 = len(nums2)
        if idx1 == len1 - 1:
            answer = nums2[idx2 + 1]
        elif idx2 == len2 - 1:
            answer = nums1[idx1 + 1]
        else:
            answer = min(nums1[idx1 + 1], nums2[idx2 + 1])
        return answer

    @staticmethod
    def simple_median(nums: List[int]) -> float:
        l = len(nums)
        if l % 2 == 0:
            half = int(l / 2)
            return (nums[half - 1] + nums[half]) / 2
        else:
            return nums[int((l - 1) / 2)]

    @staticmethod
    def find_target_index(nums: List[int], target: float) -> int:
        """
        Given a sorted array and a target, find an index in the array where the value matches the target
        (there can be many).
        Note the target can be a float in which case there cannot be a match.
        If no match is found, return the index where the value is the first one to be above the target.
        If even that does not exist, i.e. the target is larger than all in the array
        then return the length of the array
        """
        l = len(nums)
        start = 0
        end = l - 1
        found = None
        while end - start > 1 and found is None:
            mid = int((end + start) / 2)
            at_mid = nums[mid]
            if at_mid == target:
                found = mid
            elif at_mid < target:
                start = mid
            else:
                end = mid

        if found is None:
            if nums[start] >= target:
                found = start
            elif nums[end] >= target:
                found = end
            else:
                found = l

        return found
