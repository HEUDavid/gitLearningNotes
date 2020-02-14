# 剑指offer
'''
链表的环
给一个链表，若其中包含环，请找出该链表的环的入口结点，否则，输出null。
'''

# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None


class Solution:
    def EntryNodeOfLoop(self, pHead):

        meetNode = self.MeetNode(pHead)
        if not meetNode:
            return None
        lenOfLoop = self.CountNode(meetNode)
        p1 = pHead
        for i in range(lenOfLoop):
            p1 = p1.next
        p2 = pHead
        while p2 != p1:
            p2 = p2.next
            p1 = p1.next
        return p2

    def MeetNode(self, head):
        if not head:
            return None
        slow = head.next
        if slow is None:
            return None
        fast = slow.next
        while fast:
            if slow == fast:
                return slow
            slow = slow.next
            fast = fast.next.next

    def CountNode(self, meetNode):
        meetSecond = meetNode.next
        count = 1
        while meetSecond != meetNode:
            meetSecond = meetSecond.next
            count += 1
        return count


'''
在一个二维数组中（每个一维数组的长度相同），
每一行都按照从左到右递增的顺序排序，
每一列都按照从上到下递增的顺序排序。
请完成一个函数，输入这样的一个二维数组和一个整数，判断数组中是否含有该整数。
'''


class Solution:
    # array 二维列表
    def Find(self, target, array):

        lst = []
        for i in range(len(array)):
            lst += array[i]
        if target in lst:
            return True
        else:
            return False


'''
请实现一个函数，
将一个字符串中的每个空格替换成“%20”。
例如，当字符串为We Are Happy.
则经过替换之后的字符串为We%20Are%20Happy.
'''


class Solution:
    # s 源字符串
    def replaceSpace(self, s):

        s = s.replace(' ', '%20')
        return s


'''
输入一个链表，按链表值从尾到头的顺序返回一个ArrayList。
'''

# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None


class Solution:
    # 返回从尾部到头部的列表值序列，例如[1,2,3]
    def printListFromTailToHead(self, listNode):

        res = []
        while listNode:
            res.append(listNode.val)
            listNode = listNode.next
        res.reverse()
        return res


'''
重建二叉树
输入某二叉树的前序遍历和中序遍历的结果，请重建出该二叉树。
假设输入的前序遍历和中序遍历的结果中都不含重复的数字。
例如输入前序遍历序列{1,2,4,7,3,5,6,8}和中序遍历序列{4,7,2,1,5,3,8,6}，则重建二叉树并返回。
'''

# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None


class Solution:
    # 返回构造的TreeNode根节点
    def reConstructBinaryTree(self, pre, tin):

        if not pre or not tin:
            return None
        root = TreeNode(pre.pop(0))
        index = tin.index(root.val)
        root.left = self.reConstructBinaryTree(pre, tin[:index])
        root.right = self.reConstructBinaryTree(pre, tin[index + 1:])
        return root


'''
用两个栈来实现一个队列，完成队列的Push和Pop操作。队列中的元素为int类型。
'''


class Solution:
    def __init__(self):
        self.stack1 = []
        self.stack2 = []

    def push(self, node):

        self.stack1.append(node)

    def pop(self):
        # return xx
        if self.stack2:
            return self.stack2.pop(0)
        else:
            if self.stack1:
                for i in range(len(self.stack1)):
                    self.stack2.append(self.stack1.pop(0))
                return self.pop()
            else:
                return None


'''
把一个数组最开始的若干个元素搬到数组的末尾，我们称之为数组的旋转。
输入一个非减排序的数组的一个旋转，输出旋转数组的最小元素。
例如数组{3,4,5,1,2}为{1,2,3,4,5}的一个旋转，该数组的最小值为1。
NOTE：给出的所有元素都大于0，若数组大小为0，请返回0。
'''


class Solution:
    def minNumberInRotateArray(self, rotateArray):

        if not rotateArray:
            return 0
        return min(rotateArray)  # 对不起，我是开挂玩家。


class Solution:
    def minNumberInRotateArray(self, rotateArray):

        length = len(rotateArray)
        if length == 0:
            return 0
        for i in range(length):
            if rotateArray[i] > rotateArray[i + 1]:
                return rotateArray[i + 1]


'''
大家都知道斐波那契数列，现在要求输入一个整数n，请你输出斐波那契数列的第n项（从0开始，第0项为0）。
n<=39
'''


class Solution:
    def Fibonacci(self, n):

        if n == 0:
            return 0
        elif n == 1:
            return 1
        elif n > 39:
            return None
        dp = [i for i in range(n + 1)]
        for i in range(2, n + 1):
            dp[i] = dp[i - 1] + dp[i - 2]
        return dp[n]


'''
一只青蛙一次可以跳上1级台阶，也可以跳上2级。
求该青蛙跳上一个n级的台阶总共有多少种跳法（先后次序不同算不同的结果）。
'''

# 当 number == 5 时
# 从第4个台阶跳到第5个台阶，一步跳，因此有多少种方案跳到第4个台阶，有jumpFloor(4)
# 从第3个台阶跳到第5个台阶，两步跳，因此有多少种方案跳到第3个台阶，有jumpFloor(3)
# 因此，jumpFloor(5) = jumpFloor(4) + jumpFloor(3) 斐波那契数列


class Solution:
    def jumpFloor(self, number):

        dp = [1, 2]
        while len(dp) < number:
            dp.append(dp[-1] + dp[-2])
        return dp[number - 1]


'''
一只青蛙一次可以跳上1级台阶，也可以跳上2级……它也可以跳上n级。
求该青蛙跳上一个n级的台阶总共有多少种跳法。
'''

# f(n) = f(n-1) + f(n-2) + ... + f(1) + f(0)


class Solution:
    def jumpFloorII(self, number):

        dp = [1, 1]
        while len(dp) <= number:
            dp.append(sum(dp))
        return dp[number]


'''
我们可以用2*1的小矩形横着或者竖着去覆盖更大的矩形。
请问用n个2*1的小矩形无重叠地覆盖一个2*n的大矩形，总共有多少种方法？
'''

# f(n) = f(n-1) + f(n-2)


class Solution:
    def rectCover(self, number):

        if number == 0:
            return 0
        dp = [1, 2]
        while len(dp) < number:
            dp.append(dp[-1] + dp[-2])
        return dp[number - 1]


'''
输入一个整数，输出该数二进制表示中1的个数。其中负数用补码表示。
'''


class Solution:
    def NumberOf1(self, n):

        if n < 0:
            n = n + 2**32
        strN = bin(n)
        return strN.count('1')


'''
给定一个double类型的浮点数base和int类型的整数exponent。求base的exponent次方。
'''


class Solution:
    def Power(self, base, exponent):

        return base**exponent


'''
输入一个整数数组，
实现一个函数来调整该数组中数字的顺序，
使得所有的奇数位于数组的前半部分，所有的偶数位于数组的后半部分，
并保证奇数和奇数，偶数和偶数之间的相对位置不变。
'''


class Solution:
    def reOrderArray(self, array):

        odd = []
        even = []
        for i in array:
            if i % 2 == 0:
                even.append(i)
            else:
                odd.append(i)
        return odd + even


'''
输入一个链表，输出该链表中倒数第k个结点。
'''

# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None


class Solution:
    def FindKthToTail(self, head, k):

        total = self.CountNode(head)
        n = total - k
        if n < 0:
            return None
        while n > 0:
            head = head.next
            n -= 1
        return head

    def CountNode(self, head):
        count = 0
        while head:
            head = head.next
            count += 1
        return count
# 另一种方法


class Solution:
    def FindKthToTail(self, head, k):

        p1 = head
        p2 = head
        for i in range(k):
            if p2:
                p2 = p2.next
            else:
                return None
        while p2:
            p2 = p2.next
            p1 = p1.next
        return p1


'''
输入一个链表，反转链表后，输出新链表的表头。
'''

# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None


class Solution:
    # 返回ListNode
    def ReverseList(self, pHead):
        # 增加表头节点
        HeadList = ListNode(0)
        HeadList.next = pHead
        HeadReverseList = ListNode(0)
        while HeadList.next:
            q = self.DeleteNode(HeadList)
            q.next = HeadReverseList.next
            HeadReverseList.next = q
        return HeadReverseList.next

    def DeleteNode(self, link):
        p = link.next
        link.next = p.next
        return p


'''
输入两个单调递增的链表，
输出两个链表合成后的链表，当然我们需要合成后的链表满足单调不减规则。
'''

# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None


class Solution:
    # 返回合并后列表
    def Merge(self, pHead1, pHead2):

        temp = []
        while pHead1:
            temp.append(pHead1.val)
            pHead1 = pHead1.next
        while pHead2:
            temp.append(pHead2.val)
            pHead2 = pHead2.next
        temp.sort()  # 抱歉，我是开挂玩家。
        pHead3 = ListNode(0)
        p3 = pHead3
        for i in temp:
            p3.next = ListNode(i)
            p3 = p3.next
        return pHead3.next


# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None
class Solution:
    # 返回合并后列表
    def Merge(self, pHead1, pHead2):

        pHead3 = ListNode(0)
        p1 = pHead1
        p2 = pHead2
        p3 = pHead3
        while p1 and p2:
            if p1.val < p2.val:
                p3.next = p1
                p3 = p1
                p1 = p1.next
            else:
                p3.next = p2
                p3 = p2
                p2 = p2.next
        if p1:
            p3.next = p1
        else:
            p3.next = p2
        return pHead3.next


'''
输入两棵二叉树A，B，判断B是不是A的子结构。（ps：我们约定空树不是任意一个树的子结构）
'''

# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None


class Solution:
    def HasSubtree(self, pRoot1, pRoot2):

        result = False
        if pRoot1 is not None and pRoot2 is not None:
            if pRoot1.val == pRoot2.val:
                result = self.DoesTree1haveTree2(pRoot1, pRoot2)
            if not result:
                result = self.HasSubtree(pRoot1.left, pRoot2)
            if not result:
                result = self.HasSubtree(pRoot1.right, pRoot2)
        return result
    # 用于递归判断树的每个节点是否相同
    # 需要注意的地方是: 前两个if语句不可以颠倒顺序
    # 如果颠倒顺序, 会先判断pRoot1是否为None, 其实这个时候pRoot2的结点已经遍历完成确定相等了, 但是返回了False, 判断错误

    def DoesTree1haveTree2(self, pRoot1, pRoot2):
        if pRoot2 is None:
            return True
        if pRoot1 is None:
            return False
        if pRoot1.val != pRoot2.val:
            return False
        return self.DoesTree1haveTree2(
            pRoot1.left, pRoot2.left) and self.DoesTree1haveTree2(
            pRoot1.right, pRoot2.right)


'''
操作给定的二叉树，将其变换为源二叉树的镜像。
'''

# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None


class Solution:
    # 返回镜像树的根节点
    def Mirror(self, root):

        if root:
            temp = root.left
            root.left = root.right
            root.right = temp
            self.Mirror(root.left)
            self.Mirror(root.right)
        return None


'''
输入一个矩阵，按照从外向里以顺时针的顺序依次打印出每一个数字，
例如，如果输入如下4 X 4矩阵： 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16
则依次打印出数字1,2,3,4,8,12,16,15,14,13,9,5,6,7,11,10
'''


class Solution:
    # matrix类型为二维列表，需要返回列表
    def printMatrix(self, matrix):

        res = []
        while(matrix):
            res += matrix.pop(0)
            if not matrix or not matrix[0]:
                break
            matrix = self.turn(matrix)
        return res

    def turn(self, matrix):
        num_r = len(matrix)
        num_c = len(matrix[0])
        newmat = []
        for i in range(num_c):
            newmat2 = []
            for j in range(num_r):
                newmat2.append(matrix[j][i])
            newmat.append(newmat2)
        newmat.reverse()
        return newmat


'''
定义栈的数据结构，请在该类型中实现一个能够得到栈中所含最小元素的min函数（时间复杂度应为O（1））。
'''


class Solution:
    def __init__(self):
        self.array = []

    def push(self, node):

        self.array.append(node)

    def pop(self):

        self.array.pop()

    def top(self):

        return self.array[-1]

    def min(self):

        return min(self.array)

# 辅助数组


class Solution:
    def __init__(self):
        self.stack = []
        self.minList = []

    def push(self, node):

        self.stack.append(node)
        if not self.minList:
            self.minList.append(node)
        else:
            self.minList.append(min(self.minList[-1], node))

    def pop(self):

        if not self.stack:
            return None
        self.minList.pop()
        return self.stack.pop()

    def top(self):

        if not self.stack:
            return None
        return self.stack[-1]

    def min(self):

        if not self.minList:
            return None
        return self.minList[-1]


'''
输入两个整数序列，
第一个序列表示栈的压入顺序，请判断第二个序列是否可能为该栈的弹出顺序。
假设压入栈的所有数字均不相等。例如序列1,2,3,4,5是某栈的压入顺序，序列4,5,3,2,1是该压栈序列对应的一个弹出序列，
但4,3,5,1,2就不可能是该压栈序列的弹出序列。（注意：这两个序列的长度是相等的）
'''

'''
层序遍历
从上往下打印出二叉树的每个节点，同层节点从左至右打印。
'''

# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None


class Solution:
    # 返回从上到下每个节点值列表，例：[1,2,3]
    def PrintFromTopToBottom(self, root):
        if not root:
            return []
        queue = []
        res = []

        queue.append(root)
        while queue:
            node = queue.pop(0)
            res.append(node.val)
            if node.left is not None:
                queue.append(node.left)
            if node.right is not None:
                queue.append(node.right)
        return res
