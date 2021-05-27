from typing import List
import collections


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

    def __str__(self):
        return str(self.val) if self.next is None else str(self.next) + str(self.val)


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

    def __str__(self):
        if self.left is None and self.right is None:
            return f"[{self.val}]"
        left_str = f"{'None' if self.left is None else str(self.left)}"
        right_str = f"{'None' if self.right is None else str(self.right)}"
        return f"[{self.val}, {left_str}, {right_str}]"


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
        # initially we have no numbers and the sum is 0
        cum_sum[0] = -1
        cur = 0
        answer = 0
        for i, v in enumerate(nums):
            cur = cur + (1 if v == 1 else -1)
            if cur in cum_sum:
                # if the current cum sum is seen before, then we have a candidate
                # subarray, and so we just see if it is better than the current answer
                last_i = cum_sum[cur]
                answer = max(answer, i - last_i)
            else:
                cum_sum[cur] = i
        return answer

    def buildTree(self, preorder: List[int], inorder: List[int]) -> TreeNode:
        """
        Given two integer arrays preorder and inorder where preorder is the preorder traversal of a binary tree and
        inorder is the inorder traversal of the same tree, construct and return the binary tree.
        We can assume that all nodes have distinct values.
        """
        len_tree = len(preorder)
        len_tree1 = len(inorder)
        if len_tree1 != len_tree:
            raise RuntimeError(f"unmatched lengths ${len_tree} ${len_tree1}")
        if not preorder:
            return None
        root = preorder[0]
        if len_tree == 1:
            return TreeNode(root)
        for i, v in enumerate(inorder):
            if v == root:
                root_inorder_idx = i
                break
        if root_inorder_idx == 0:
            left = None
        else:
            # print("rii", root_inorder_idx, preorder[1: root_inorder_idx])
            left = self.buildTree(preorder[1: root_inorder_idx + 1], inorder[0: root_inorder_idx])
        if root_inorder_idx == len_tree - 1:
            right = None
        else:
            right = self.buildTree(preorder[root_inorder_idx + 1:], inorder[root_inorder_idx + 1:])
        return TreeNode(root, left, right)

    def minimumTotal(self, triangle: List[List[int]]) -> int:
        """
        Given a triangle array, return the minimum path sum from top to bottom.
        """
        calculated = dict()  # (row, col) to min_total
        return Solution.minimum_total(triangle, 0, 0, calculated)

    @staticmethod
    def minimum_total(triangle: List[List[int]], row: int, col: int, calculated: dict) -> int:
        row_col = (row, col)
        if row_col in calculated:
            return calculated[row_col]
        num_rows = len(triangle) - row
        val_at = triangle[row][col]
        if num_rows == 1:
            min_total = val_at
        else:
            min_total = val_at + min(Solution.minimum_total(triangle, row + 1, col, calculated),
                                     Solution.minimum_total(triangle, row + 1, col + 1, calculated))
        calculated[row_col] = min_total
        return min_total

    def evalRPN(self, tokens: List[str]) -> int:
        """
        HP calculator: https://leetcode.com/problems/evaluate-reverse-polish-notation/
        """
        dq = collections.deque()
        for t in tokens:
            try:
                dq.append(int(t))
            except Exception:
                right = dq.pop()
                left = dq.pop()
                if t == '+':
                    result = left + right
                elif t == '-':
                    result = left - right
                elif t == '*':
                    result = left * right
                elif t == '/':
                    result = int(left / right)
                else:
                    raise RuntimeError(t)
                dq.append(result)
        return dq.pop()

    def evalRPN1(self, tokens: List[str]) -> int:
        # slow answer for https://leetcode.com/problems/evaluate-reverse-polish-notation/, too much recursion
        # the last one must be an operator, the job is just to find the two operands
        operator = tokens[-1]
        if operator == '+':
            left, right = self.find_operands(tokens)
            return self.evalRPN(left) + self.evalRPN(right)
        elif operator == '-':
            left, right = self.find_operands(tokens)
            return self.evalRPN(left) - self.evalRPN(right)
        elif operator == '*':
            left, right = self.find_operands(tokens)
            return self.evalRPN(left) * self.evalRPN(right)
        elif operator == '/':
            left, right = self.find_operands(tokens)
            return int(self.evalRPN(left) / self.evalRPN(right))
        else:
            return int(operator)

    def find_operands(self, tokens: List[str]):
        int_count = 0
        oper_count = 0
        l = len(tokens)
        for i in range(l - 2, 0, -1):
            try:
                int(tokens[i])
                int_count = int_count + 1
            except Exception:
                oper_count = oper_count + 1
            print(i, int_count, oper_count)
            if int_count - oper_count == 1:
                return tokens[0: i], tokens[i: l - 1]
        raise RuntimeError("invalid tokens" + str(tokens))
