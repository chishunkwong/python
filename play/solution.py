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
        start1 = 0
        start2 = 0
        half_index1 = None
        half_index2 = None
        use_one = None
        while half_index1 is None and half_index2 is None:
            need_ahead = half - start1 - start2 - 2
            if need_ahead == 0:
                one_more1 = None if start1 + 1 == len1 else nums1[start1 + 1]
                one_more2 = None if start2 + 1 == len2 else nums2[start2 + 1]
                # TBC, we would have start1 = 0 and start2 = 0
                if one_more1 is None or one_more2 is not None and one_more1 > one_more2:
                    half_index1 = start1
                    half_index2 = start2 + 1
                    use_one = False
                else:
                    half_index2 = start2
                    half_index1 = start1 + 1
                    use_one = True
                break
            # now we try to find the element at the index half (within the combined, sorted, array)
            half_need = int(need_ahead / 2)
            if len1 - start1 <= len2 - start2:
                # first array has fewer numbers (counting from the current start), so we take
                # from it first as it is the one that may not have enough elements to walk ahead of
                take1 = half_need if half_need <= len1 else len1 - start1
                take2 = need_ahead - take1
            else:
                # vice versa
                take2 = half_need if half_need <= len2 else len2 - start2
                take1 = need_ahead - take2
            at1 = nums1[start1 + take1 - 1]
            at2 = nums2[start2 + take2 - 1]
            print(at1, at2)
            if at1 == at2:
                # we got lucky, the two arrays have the same value, so it does not matter which one we take from
                # and it will still give us the half value, we will use nums1
                half_index1 = start1 + take1 - 1
                half_index2 = start2 + take2 - 1
                use_one = True
                break
            if at1 > at2:
                # in case nums2 is actually exhausted, then it is easy
                if start2 + take2 == len2:
                    half_index1 = half - len2 - 1
                    half_index2 = None
                    use_one = True
                else:
                    # we took two samples, one from each array, and found that the one at array 1 is larger
                    # so we step back in array 1. I.e. we change the start1 to match at2, and try again
                    start1 = Solution.find_target_index(nums1, at2)
                    start2 = start2 + take2
            else:
                # must be at2 > at1, because we already dealt with the equal case
                # do the same thing as above, just 1 is 2 and 2 is 1
                if start1 + take1 == len1:
                    half_index2 = half - len1 - 1
                    half_index1 = None
                    use_one = False
                else:
                    start2 = Solution.find_target_index(nums2, at1)
                    start1 = start1 + take1
        if use_one:
            median = nums1[half_index1]
        else:
            median = nums2[half_index2]

        # whew, now we know what's at half, but still need to do the even case
        if not is_odd:
            just_below = median
            if half_index1 is None:
                just_above = nums2[half_index2 + 1]
            elif half_index2 is None:
                just_above = nums1[half_index1 + 1]
            else:
                ja1 = None if half_index1 + 1 == len1 else nums1[half_index1 + 1]
                ja2 = None if half_index2 + 1 == len2 else nums2[half_index2 + 1]
                if ja1 is not None and ja2 is not None and ja1 >= just_below and ja2 >= just_below:
                    just_above = min(ja1, ja2)
                elif ja1 is None or ja1 < just_below:
                    just_above = ja2
                else:
                    just_above = ja1
            median = (just_below + just_above) / 2

        return median

    @staticmethod
    def simple_median(nums: List[int]) -> float:
        l = len(nums)
        if l % 2 == 0:
            return (nums[l / 2 - 1] + nums[l / 2]) / 2
        else:
            return nums[(l + 1) / 2]

    @staticmethod
    def find_target_index(nums: List[int], target: int) -> int:
        """
        Given a sorted array and a target, find an index in the array where the value matches the target,
        if no match is found, return the index where the value is the last one to be below the target.
        If even that does not exist, i.e. the target is smaller than even the first value, then still return 0
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
                start = mid + 1
            else:
                end = mid - 1
        if found is None:
            if nums[start] >= target:
                return start
            if nums[end] <= target:
                return end
            return start
        return found
