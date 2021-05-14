from typing import List


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

    def __str__(self):
        return str(self.val) if self.next is None else str(self.next) + str(self.val)


class Solution:

    def addTwoNumbers(self, l1: ListNode, l2: ListNode) -> ListNode:
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
        return 0
