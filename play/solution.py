from typing import List


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

    def __str__(self):
        return str(self.val) if self.next is None else str(self.next) + str(self.val)


class Solution:

    def addTwoNumbers(self, l1: ListNode, l2: ListNode) -> ListNode:
        """
        Add two numbers, where each number is represented as a linked list with the digits
        in reverse order, so for example 1 -> 2 -> 3 is 321.
        Return the sum in the same format
        """
        if l1 is None and l2 is None:
            return None
        if l1 is None:
            return l2
        if l2 is None:
            return l1
        digit_sum = l1.val + l2.val
        if digit_sum < 10:
            answer = ListNode(digit_sum, self.addTwoNumbers(l1.next, l2.next))
        else:
            answer = ListNode(digit_sum % 10, self.addTwoNumbers(self.addTwoNumbers(ListNode(1), l1.next), l2.next))
        return answer

    def findMaxLength(self, nums: List[int]) -> int:
        """
        Given a binary array nums, return the maximum length of a contiguous subarray
        with an equal number of 0 and 1.
        """
        cum_sum = dict()
        cum_sum[0] = -1
        cur = 0
        answer = 0
        for i, v in enumerate(nums):
            cur = cur + (1 if v == 1 else -1)
            if cur in cum_sum:
                last_i = cum_sum[cur]
                answer = max(answer, i - last_i)
            else:
                cum_sum[cur] = i
        return answer
