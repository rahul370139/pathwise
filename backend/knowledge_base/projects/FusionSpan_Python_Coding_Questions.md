# FusionSpan Interview Prep — Python Coding Questions for Data Engineering

**Candidate:** Rahul Sharma | **Target:** Data Engineer / Data Integration Specialist — FusionSpan  
**Focus:** Data Structures, Algorithms, Python Patterns for ETL, Linked Lists, Stacks, Queues, Recursion, String/Array Manipulation  
**Document Scope:** 30+ curated coding problems likely asked at data engineering interviews — with solutions, complexity analysis, and interview tips

---

## Table of Contents

1. [Arrays & Lists](#1-arrays--lists)
2. [Strings](#2-strings)
3. [Linked Lists](#3-linked-lists)
4. [Stacks & Queues](#4-stacks--queues)
5. [Recursion](#5-recursion)
6. [Dictionaries & Hash Maps](#6-dictionaries--hash-maps)
7. [Sorting & Searching](#7-sorting--searching)
8. [Data Engineering Python Patterns](#8-data-engineering-python-patterns)
9. [SQL-in-Python (Pandas)](#9-sql-in-python-pandas)
10. [Complexity Cheat Sheet](#10-complexity-cheat-sheet)

---

# 1. Arrays & Lists

---

### Q1: Two Sum

Given an array and a target, return indices of two numbers that add up to target.

```python
def two_sum(nums: list[int], target: int) -> list[int]:
    seen = {}
    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:
            return [seen[complement], i]
        seen[num] = i
    return []

assert two_sum([2, 7, 11, 15], 9) == [0, 1]
assert two_sum([3, 2, 4], 6) == [1, 2]
# Time: O(n), Space: O(n)
```

---

### Q2: Remove Duplicates from Sorted Array (In-Place)

```python
def remove_duplicates(nums: list[int]) -> int:
    if not nums:
        return 0
    write = 1
    for read in range(1, len(nums)):
        if nums[read] != nums[read - 1]:
            nums[write] = nums[read]
            write += 1
    return write

nums = [1, 1, 2, 2, 3]
k = remove_duplicates(nums)
assert nums[:k] == [1, 2, 3]
# Time: O(n), Space: O(1)
```

---

### Q3: Maximum Subarray Sum (Kadane's Algorithm)

```python
def max_subarray(nums: list[int]) -> int:
    max_sum = current = nums[0]
    for num in nums[1:]:
        current = max(num, current + num)
        max_sum = max(max_sum, current)
    return max_sum

assert max_subarray([-2, 1, -3, 4, -1, 2, 1, -5, 4]) == 6  # [4, -1, 2, 1]
# Time: O(n), Space: O(1)
```

---

### Q4: Rotate Array by K Steps

```python
def rotate(nums: list[int], k: int) -> None:
    n = len(nums)
    k = k % n
    def reverse(start, end):
        while start < end:
            nums[start], nums[end] = nums[end], nums[start]
            start += 1
            end -= 1
    reverse(0, n - 1)
    reverse(0, k - 1)
    reverse(k, n - 1)

nums = [1, 2, 3, 4, 5, 6, 7]
rotate(nums, 3)
assert nums == [5, 6, 7, 1, 2, 3, 4]
# Time: O(n), Space: O(1)
```

---

### Q5: Find Missing Number (0 to n)

```python
def missing_number(nums: list[int]) -> int:
    n = len(nums)
    expected = n * (n + 1) // 2
    return expected - sum(nums)

assert missing_number([3, 0, 1]) == 2
assert missing_number([0, 1]) == 2
# Time: O(n), Space: O(1)
```

---

### Q6: Merge Two Sorted Arrays

```python
def merge_sorted(a: list[int], b: list[int]) -> list[int]:
    result = []
    i = j = 0
    while i < len(a) and j < len(b):
        if a[i] <= b[j]:
            result.append(a[i])
            i += 1
        else:
            result.append(b[j])
            j += 1
    result.extend(a[i:])
    result.extend(b[j:])
    return result

assert merge_sorted([1, 3, 5], [2, 4, 6]) == [1, 2, 3, 4, 5, 6]
# Time: O(n + m), Space: O(n + m)
```

---

### Q7: Move Zeroes to End (In-Place)

```python
def move_zeroes(nums: list[int]) -> None:
    write = 0
    for read in range(len(nums)):
        if nums[read] != 0:
            nums[write], nums[read] = nums[read], nums[write]
            write += 1

nums = [0, 1, 0, 3, 12]
move_zeroes(nums)
assert nums == [1, 3, 12, 0, 0]
# Time: O(n), Space: O(1)
```

---

# 2. Strings

---

### Q8: Reverse a String (In-Place)

```python
def reverse_string(s: list[str]) -> None:
    left, right = 0, len(s) - 1
    while left < right:
        s[left], s[right] = s[right], s[left]
        left += 1
        right -= 1
# Time: O(n), Space: O(1)
```

---

### Q9: Check if Two Strings are Anagrams

```python
from collections import Counter

def is_anagram(s: str, t: str) -> bool:
    return Counter(s) == Counter(t)

assert is_anagram("listen", "silent") == True
assert is_anagram("hello", "world") == False
# Time: O(n), Space: O(1) — bounded alphabet
```

---

### Q10: Longest Substring Without Repeating Characters

```python
def longest_unique_substring(s: str) -> int:
    seen = {}
    left = 0
    max_len = 0
    for right, char in enumerate(s):
        if char in seen and seen[char] >= left:
            left = seen[char] + 1
        seen[char] = right
        max_len = max(max_len, right - left + 1)
    return max_len

assert longest_unique_substring("abcabcbb") == 3  # "abc"
assert longest_unique_substring("bbbbb") == 1
# Time: O(n), Space: O(min(n, alphabet_size))
```

---

### Q11: Valid Palindrome (Alphanumeric Only)

```python
def is_palindrome(s: str) -> bool:
    cleaned = [c.lower() for c in s if c.isalnum()]
    return cleaned == cleaned[::-1]

assert is_palindrome("A man, a plan, a canal: Panama") == True
assert is_palindrome("race a car") == False
# Time: O(n), Space: O(n)
```

---

### Q12: Count Vowels and Consonants

```python
def count_vowels_consonants(s: str) -> dict:
    vowels = set("aeiouAEIOU")
    v = c = 0
    for ch in s:
        if ch.isalpha():
            if ch in vowels:
                v += 1
            else:
                c += 1
    return {"vowels": v, "consonants": c}
```

---

# 3. Linked Lists

**This is a reported FusionSpan interview question.** Practice reverse a linked list thoroughly.

---

### Q13: Define a Linked List Node

```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def to_list(head: ListNode) -> list:
    """Helper to convert linked list to Python list for testing."""
    result = []
    while head:
        result.append(head.val)
        head = head.next
    return result

def from_list(vals: list) -> ListNode:
    """Helper to create linked list from Python list."""
    dummy = ListNode(0)
    curr = dummy
    for v in vals:
        curr.next = ListNode(v)
        curr = curr.next
    return dummy.next
```

---

### Q14: Reverse a Linked List (Iterative) — MUST KNOW

```python
def reverse_list(head: ListNode) -> ListNode:
    prev = None
    curr = head
    while curr:
        next_node = curr.next   # save next
        curr.next = prev        # reverse pointer
        prev = curr             # advance prev
        curr = next_node        # advance curr
    return prev

# Walk-through: 1 → 2 → 3 → None
# Step 1: prev=None, curr=1 → next=2, 1.next=None, prev=1, curr=2
# Step 2: prev=1, curr=2 → next=3, 2.next=1, prev=2, curr=3
# Step 3: prev=2, curr=3 → next=None, 3.next=2, prev=3, curr=None
# Result: 3 → 2 → 1 → None

head = from_list([1, 2, 3, 4, 5])
assert to_list(reverse_list(head)) == [5, 4, 3, 2, 1]
# Time: O(n), Space: O(1)
```

---

### Q15: Reverse a Linked List (Recursive)

```python
def reverse_list_recursive(head: ListNode) -> ListNode:
    if not head or not head.next:
        return head
    new_head = reverse_list_recursive(head.next)
    head.next.next = head
    head.next = None
    return new_head
# Time: O(n), Space: O(n) — call stack
```

---

### Q16: Detect Cycle in a Linked List (Floyd's)

```python
def has_cycle(head: ListNode) -> bool:
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow is fast:
            return True
    return False
# Time: O(n), Space: O(1) — two-pointer technique
```

---

### Q17: Find Middle of Linked List

```python
def find_middle(head: ListNode) -> ListNode:
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
    return slow

head = from_list([1, 2, 3, 4, 5])
assert find_middle(head).val == 3
# Time: O(n), Space: O(1)
```

---

### Q18: Merge Two Sorted Linked Lists

```python
def merge_lists(l1: ListNode, l2: ListNode) -> ListNode:
    dummy = ListNode(0)
    curr = dummy
    while l1 and l2:
        if l1.val <= l2.val:
            curr.next = l1
            l1 = l1.next
        else:
            curr.next = l2
            l2 = l2.next
        curr = curr.next
    curr.next = l1 or l2
    return dummy.next
# Time: O(n + m), Space: O(1)
```

---

### Q19: Remove Nth Node from End

```python
def remove_nth_from_end(head: ListNode, n: int) -> ListNode:
    dummy = ListNode(0, head)
    fast = slow = dummy
    for _ in range(n + 1):
        fast = fast.next
    while fast:
        slow = slow.next
        fast = fast.next
    slow.next = slow.next.next
    return dummy.next
# Time: O(n), Space: O(1)
```

---

# 4. Stacks & Queues

---

### Q20: Valid Parentheses

```python
def is_valid(s: str) -> bool:
    stack = []
    pairs = {")": "(", "]": "[", "}": "{"}
    for ch in s:
        if ch in "({[":
            stack.append(ch)
        elif ch in ")}]":
            if not stack or stack[-1] != pairs[ch]:
                return False
            stack.pop()
    return len(stack) == 0

assert is_valid("()[]{}") == True
assert is_valid("([)]") == False
assert is_valid("{[()]}") == True
# Time: O(n), Space: O(n)
```

---

### Q21: Implement Queue Using Two Stacks

```python
class QueueFromStacks:
    def __init__(self):
        self.stack_in = []
        self.stack_out = []

    def enqueue(self, val):
        self.stack_in.append(val)

    def dequeue(self):
        if not self.stack_out:
            while self.stack_in:
                self.stack_out.append(self.stack_in.pop())
        return self.stack_out.pop() if self.stack_out else None

    def peek(self):
        if not self.stack_out:
            while self.stack_in:
                self.stack_out.append(self.stack_in.pop())
        return self.stack_out[-1] if self.stack_out else None
# Amortized O(1) per operation
```

---

### Q22: Min Stack (O(1) getMin)

```python
class MinStack:
    def __init__(self):
        self.stack = []
        self.min_stack = []

    def push(self, val: int):
        self.stack.append(val)
        if not self.min_stack or val <= self.min_stack[-1]:
            self.min_stack.append(val)

    def pop(self):
        val = self.stack.pop()
        if val == self.min_stack[-1]:
            self.min_stack.pop()
        return val

    def get_min(self) -> int:
        return self.min_stack[-1]
```

---

### Q23: Implement Stack Using Queue (collections.deque)

```python
from collections import deque

class StackFromQueue:
    def __init__(self):
        self.q = deque()

    def push(self, val):
        self.q.append(val)
        for _ in range(len(self.q) - 1):
            self.q.append(self.q.popleft())

    def pop(self):
        return self.q.popleft()

    def top(self):
        return self.q[0]
```

---

# 5. Recursion

---

### Q24: Fibonacci (with Memoization)

```python
from functools import lru_cache

@lru_cache(maxsize=None)
def fib(n: int) -> int:
    if n < 2:
        return n
    return fib(n - 1) + fib(n - 2)

assert fib(10) == 55
assert fib(0) == 0
# Time: O(n) with memoization, Space: O(n)
```

---

### Q25: Factorial

```python
def factorial(n: int) -> int:
    if n < 0:
        raise ValueError("n must be non-negative")
    if n <= 1:
        return 1
    return n * factorial(n - 1)

assert factorial(5) == 120
assert factorial(0) == 1
```

---

### Q26: Flatten Nested List (Recursive)

```python
def flatten(nested) -> list:
    result = []
    for item in nested:
        if isinstance(item, (list, tuple)):
            result.extend(flatten(item))
        else:
            result.append(item)
    return result

assert flatten([1, [2, [3, 4], 5], [6, 7]]) == [1, 2, 3, 4, 5, 6, 7]
# Time: O(total elements), Space: O(depth) call stack
```

---

### Q27: Power Function (Recursive, O(log n))

```python
def power(base: float, exp: int) -> float:
    if exp == 0:
        return 1
    if exp < 0:
        return 1 / power(base, -exp)
    if exp % 2 == 0:
        half = power(base, exp // 2)
        return half * half
    return base * power(base, exp - 1)

assert power(2, 10) == 1024
assert power(2, -2) == 0.25
# Time: O(log n)
```

---

# 6. Dictionaries & Hash Maps

---

### Q28: Group Anagrams

```python
from collections import defaultdict

def group_anagrams(words: list[str]) -> list[list[str]]:
    groups = defaultdict(list)
    for word in words:
        key = tuple(sorted(word))
        groups[key].append(word)
    return list(groups.values())

result = group_anagrams(["eat", "tea", "tan", "ate", "nat", "bat"])
# [['eat', 'tea', 'ate'], ['tan', 'nat'], ['bat']]
```

---

### Q29: First Non-Repeating Character

```python
from collections import Counter

def first_unique_char(s: str) -> int:
    counts = Counter(s)
    for i, ch in enumerate(s):
        if counts[ch] == 1:
            return i
    return -1

assert first_unique_char("leetcode") == 0  # 'l'
assert first_unique_char("aabb") == -1
# Time: O(n), Space: O(1) — bounded alphabet
```

---

### Q30: Top K Frequent Elements

```python
from collections import Counter

def top_k_frequent(nums: list[int], k: int) -> list[int]:
    return [x for x, _ in Counter(nums).most_common(k)]

assert top_k_frequent([1, 1, 1, 2, 2, 3], 2) == [1, 2]
```

---

# 7. Sorting & Searching

---

### Q31: Binary Search

```python
def binary_search(nums: list[int], target: int) -> int:
    left, right = 0, len(nums) - 1
    while left <= right:
        mid = (left + right) // 2
        if nums[mid] == target:
            return mid
        elif nums[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1

assert binary_search([1, 3, 5, 7, 9], 5) == 2
assert binary_search([1, 3, 5, 7, 9], 4) == -1
# Time: O(log n), Space: O(1)
```

---

### Q32: Sort a List of Dicts by Key

```python
records = [
    {"name": "Alice", "age": 30},
    {"name": "Bob", "age": 25},
    {"name": "Charlie", "age": 35},
]

by_age = sorted(records, key=lambda r: r["age"])
by_name = sorted(records, key=lambda r: r["name"])

from operator import itemgetter
by_age_alt = sorted(records, key=itemgetter("age"))
```

---

# 8. Data Engineering Python Patterns

These are practical questions a data engineering interviewer might ask.

---

### Q33: Read CSV and Compute Column Stats

```python
import csv
from collections import defaultdict

def csv_column_stats(filepath: str) -> dict:
    """Compute mean for numeric columns in a CSV."""
    sums = defaultdict(float)
    counts = defaultdict(int)

    with open(filepath) as f:
        reader = csv.DictReader(f)
        for row in reader:
            for key, val in row.items():
                try:
                    sums[key] += float(val)
                    counts[key] += 1
                except (ValueError, TypeError):
                    pass

    return {k: sums[k] / counts[k] for k in sums}
```

---

### Q34: Deduplicate Records by Key

```python
def deduplicate(records: list[dict], key: str) -> list[dict]:
    """Keep last occurrence of each key."""
    seen = {}
    for record in records:
        seen[record[key]] = record
    return list(seen.values())

data = [
    {"id": 1, "name": "Alice"},
    {"id": 2, "name": "Bob"},
    {"id": 1, "name": "Alice Updated"},
]
assert len(deduplicate(data, "id")) == 2
```

---

### Q35: Chunk a Large List for Batch Processing

```python
def chunk_list(data: list, chunk_size: int):
    """Yield successive chunks from data."""
    for i in range(0, len(data), chunk_size):
        yield data[i:i + chunk_size]

# Process in batches of 1000
for batch in chunk_list(large_dataset, 1000):
    process_batch(batch)
```

---

### Q36: Parse JSON API Response and Extract Nested Field

```python
def extract_emails(api_response: list[dict]) -> list[str]:
    """Extract emails from nested JSON structure."""
    emails = []
    for user in api_response:
        email = user.get("contact", {}).get("email")
        if email:
            emails.append(email)
    return emails

data = [
    {"name": "Alice", "contact": {"email": "alice@example.com", "phone": "123"}},
    {"name": "Bob", "contact": {"email": "bob@example.com"}},
    {"name": "Charlie", "contact": {}},
]
assert extract_emails(data) == ["alice@example.com", "bob@example.com"]
```

---

### Q37: Implement a Simple ETL Function

```python
def simple_etl(raw_records: list[dict]) -> list[dict]:
    """Extract, transform, and return cleaned records."""
    cleaned = []
    for record in raw_records:
        if not record.get("name") or not record.get("amount"):
            continue

        cleaned.append({
            "name": record["name"].strip().title(),
            "amount": round(float(record["amount"]), 2),
            "date": record.get("date", "unknown"),
            "is_valid": float(record["amount"]) > 0,
        })
    return cleaned
```

---

### Q38: Process a Large File Line-by-Line (Generator Pattern)

```python
def process_large_file(filepath: str):
    """Memory-efficient line-by-line processing using generators."""
    def read_lines(path):
        with open(path) as f:
            for line in f:
                yield line.strip()

    def parse_records(lines):
        for line in lines:
            parts = line.split(",")
            if len(parts) >= 3:
                yield {"id": parts[0], "name": parts[1], "value": parts[2]}

    def filter_valid(records):
        for rec in records:
            if rec["value"].replace(".", "").isdigit():
                yield rec

    lines = read_lines(filepath)
    parsed = parse_records(lines)
    valid = filter_valid(parsed)

    for record in valid:
        process(record)
```

---

# 9. SQL-in-Python (Pandas)

Interviewers may ask you to do SQL-like operations in Pandas.

### Q39: GROUP BY + Aggregation

```python
import pandas as pd

df = pd.DataFrame({
    "dept": ["Eng", "Eng", "Sales", "Sales", "Eng"],
    "salary": [90000, 85000, 70000, 75000, 95000],
    "name": ["Alice", "Bob", "Charlie", "Diana", "Eve"],
})

result = df.groupby("dept").agg(
    avg_salary=("salary", "mean"),
    max_salary=("salary", "max"),
    headcount=("name", "count"),
).reset_index()
```

### Q40: Window Function (Rank per Group)

```python
df["rank_in_dept"] = df.groupby("dept")["salary"].rank(
    ascending=False, method="dense"
)

top_earners = df[df["rank_in_dept"] == 1]
```

### Q41: Merge (JOIN)

```python
employees = pd.DataFrame({"emp_id": [1, 2, 3], "dept_id": [10, 20, 10]})
departments = pd.DataFrame({"dept_id": [10, 20, 30], "dept_name": ["Eng", "Sales", "HR"]})

joined = pd.merge(employees, departments, on="dept_id", how="left")
```

---

# 10. Complexity Cheat Sheet

| Data Structure | Access | Search | Insert | Delete |
|---------------|--------|--------|--------|--------|
| **Array/List** | O(1) | O(n) | O(n)* | O(n) |
| **Dict/HashMap** | — | O(1) avg | O(1) avg | O(1) avg |
| **Set** | — | O(1) avg | O(1) avg | O(1) avg |
| **Stack (list)** | O(n) | O(n) | O(1)* push | O(1) pop |
| **Queue (deque)** | O(n) | O(n) | O(1) append | O(1) popleft |
| **Linked List** | O(n) | O(n) | O(1) at head | O(1) at head |
| **Min-Heap** | — | O(n) | O(log n) | O(log n) |
| **Sorted Array** | O(1) | O(log n) | O(n) | O(n) |

| Algorithm | Time | Space |
|-----------|------|-------|
| Binary Search | O(log n) | O(1) |
| Merge Sort | O(n log n) | O(n) |
| Quick Sort | O(n log n) avg | O(log n) |
| BFS/DFS | O(V + E) | O(V) |
| Two Pointers | O(n) | O(1) |
| Sliding Window | O(n) | O(k) |

---

## Interview Tips

1. **Always clarify** the problem before coding. Ask about edge cases, input size, duplicates.
2. **Start with brute force**, then optimize. Say "The naive approach is O(n^2), but we can do O(n) with a hash map."
3. **Talk through your approach** before writing code. The interviewer wants to see your thought process.
4. **Test with examples** after coding. Walk through with the given example, then try edge cases (empty input, single element, duplicates).
5. **Know your complexities.** Always state time and space complexity after solving.
6. **For linked list problems:** Draw the pointers on paper. The three variables `prev`, `curr`, `next_node` pattern solves most problems.
7. **For data engineering roles:** They care more about practical patterns (file processing, dedup, batch processing) than LeetCode hards.

---

*Prepared for Rahul Sharma — Python Coding Questions for FusionSpan Data Engineering Interview.*
