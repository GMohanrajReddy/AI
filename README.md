# GM 
# DSD PROGRAM 
#####

---

**Disclaimer:**

<sup>These programs are for reference only. Do not copy them for use in assignments or labs, including the Data Structures Design (DSD) lab. Use them for learning and develop your own solutions independently. The author is not responsible for any misuse.</sup>

---
#### 1(A) Write a Python PROGRAM to calculate electricity bill for a given tariff.

• 1 to 100 units – Rs. 10

• 100 to 200 units – Rs. 15

• 200 to 300 units – Rs.20

• above 300 units – Rs. 25
#### PROGRAM
```
units = float(input("Enter the number of units consumed:"))

billamount = units * 10 if units <= 100 else (1000 + (units - 100) * 15) if units <= 200 else (2500 + (units - 200) * 20) if units <= 300 else (4500 + (units - 300) * 25)

print(f"The electricity bill for {units} units is Rs. {billamount}")
```
---
#### 1(B) To write a Python PROGRAM for basic operations of calculator.
#### PROGRAM
```
from abc import ABC, abstractmethod

class Operation(ABC):

    def operate(self, a, b):
        pass

class Addition(Operation):

    def operate(self, a, b):
        return a + b

class Subtraction(Operation):

    def operate(self, a, b):
        return a - b

class Multiplication(Operation):

    def operate(self, a, b):
        return a * b

class Division(Operation):

    def operate(self, a, b):
        if b != 0:
            return a / b
        else:
            return "Cannot divide by zero"

a = float(input("Enter the first number: "))
b = float(input("Enter the second number: "))

addition = Addition()
print("Addition:", addition.operate(a, b))

subtraction = Subtraction()
print("Subtraction:", subtraction.operate(a, b))

multiplication = Multiplication()
print("Multiplication:", multiplication.operate(a, b))

division = Division()
print("Division:", division.operate(a, b))
```
---
#### 2(A) Write a Python PROGRAM to find factorial calculation using recursion.
#### PROGRAM
```
def factorial(n):
    return 1 if n == 0 else n * factorial(n - 1)

num = int(input("Enter a number: "))
print("Factorial of", num, "is", factorial(num))
```
##
#### 2(B) To write a Python PROGRAM for Tower of Hanoi using recursion.
#### PROGRAM
```
def tower_of_hanoi(n, source, target, auxiliary):
    if n == 1:
        print(f"Move disk 1 from {source} to {target}")
        return
    tower_of_hanoi(n - 1, source, auxiliary, target)
    print(f"Move disk {n} from {source} to {target}")
    tower_of_hanoi(n - 1, auxiliary, target, source)
n = int(input("Enter the number of disks: "))
tower_of_hanoi(n, 'A', 'C', 'B')

```
---
#### 3(A) To write a Python PROGRAM to search element in a list using arrays.
#### PROGRAM
```
class ArrayList:
    def __init__(self):
        self.array = []
    def append(self, element):
        self.array.append(element)
    def search(self, element):
        for i in range(len(self.array)):
            if self.array[i] == element:
                return True, i
        return False, -1
arr_list = ArrayList()
n = int(input("Enter the number of elements in the list: "))
for _ in range(n):
    element = float(input("Enter an element: "))
    arr_list.append(element)
element_to_search = float(input("Enter the element to search: "))
found, index = arr_list.search(element_to_search)
if found:
    print(f"Element {element_to_search} found at index {index}")
else:
    print(f"Element {element_to_search} not found")

```
---
#### 4(A) To write a Python PROGRAM to create linked list with n elements.
#### PROGRAM
```
class Node:
    def __init__(self, data=None):
        self.data = data
        self.next = None

def create_linked_list():
    n = int(input("How many elements would you like to add? "))
    if n <= 0:
        return None
    head = Node(float(input("Enter data item: ")))
    current = head
    for _ in range(n - 1):
        current.next = Node(float(input("Enter data item: ")))
        current = current.next
    return head

def display_linked_list(head):
    current = head
    while current:
        print(current.data, end=" ")
        current = current.next
    print()

head = create_linked_list()
print("The linked list:", end=" ")
display_linked_list(head)
```
---
#### 4(B) To write a Python PROGRAM to search key element in a linked list.
#### PROGRAM
```
class Node:
    def __init__(self, data):
        self.data = data
        self.next = None
class LinkedList:
    def __init__(self):
        self.head = None
    def append(self, data):
        new_node = Node(data)
        if self.head is None:
            self.head = new_node
            return
        last_node = self.head
        while last_node.next:
            last_node = last_node.next
        last_node.next = new_node
    def search(self, key):
        current = self.head
        index = 0
        while current:
            if current.data == key:
                return True, index
            current = current.next
            index += 1
        return False, -1
llist = LinkedList()
for _ in range(int(input("Enter the number of elements in the linked list: "))):
    llist.append(float(input("Enter an element: ")))
key = float(input("Enter the element to search: "))
found, index = llist.search(key)
if found:
    print(f"Key element {key} found at index {index} in the linked list")
else:
    print(f"Key element {key} not found in the linked list")

```
---
#### 5(A) To write a Python PROGRAM to insert elements into stack.
#### PROGRAM
```
class Stack:
    def __init__(self):
        self.stack = []

    def push(self, element):
        self.stack.append(element)
        self.display()

    def pop(self, element=None):
        if element is None:
            if not self.is_empty():
                self.stack.pop()
                self.display()
            else:
                print("Stack is empty")
        else:
            try:
                self.stack.remove(element)
                self.display()
            except ValueError:
                print(f"Element '{element}' not found in the stack")

    def is_empty(self):
        return len(self.stack) == 0

    def display(self):
        print("Stack:", self.stack)

stack = Stack()

while True:
    action = input("Enter action (push/pop/done): ")
    if action.lower() == 'done':
        break
    elif action.lower() == 'push':
        element = input("Enter element to push: ")
        stack.push(element)
    elif action.lower() == 'pop':
        element = input("Enter element to pop (leave blank to pop the top element): ")
        stack.pop(element if element else None)
    else:
        print("Invalid action. Please enter 'push', 'pop', or 'done'")

```
---
#### 5(B) To write a Python PROGRAM to implement queue.
#### PROGRAM
```
queue = []

def enqueue(item):
    queue.append(item)

def dequeue():
    if not is_empty():
        return queue.pop(0)
    else:
        raise IndexError("dequeue from empty queue")

def is_empty():
    return not queue

def size():
    return len(queue)

def display():
    print("Queue:", queue)

def remove_element(element):
    if element in queue:
        queue.remove(element)
        print(f"Element '{element}' removed from the queue")
    else:
        print(f"Element '{element}' not found in the queue")
while True:
    action = input("Enter 'enqueue' to add an item, 'remove' to remove a specific element, or 'done' to stop: ")
    if action.lower() == 'done':
        break
    elif action.lower() == 'enqueue':
        value = input("Enter a value to enqueue: ")
        enqueue(value)
        display()
    elif action.lower() == 'remove':
        element_to_remove = input("Enter the element to remove from the queue: ")
        remove_element(element_to_remove)
        display()
    else:
        print("Invalid action")
display()
print("Size:", size())
print("Is empty:", is_empty())

```
---
#### 6(A) To write a Python PROGRAM for card of game in Python in List ADT.
#### PROGRAM
```
import random

suits = ['Hearts', 'Diamonds', 'Clubs', 'Spades']
ranks = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'Jack', 'Queen', 'King', 'Ace']
deck = [(rank, suit) for suit in suits for rank in ranks]
random.shuffle(deck)

def deal_hand(num_cards):
    return [deck.pop() for _ in range(num_cards)]

def display_hand(player, hand):
    print(f"{player}'s hand:")
    for card in hand:
        print(f"{card[0]} of {card[1]}")

def determine_winner(player_hand, computer_hand):
    player_ranks = [ranks.index(card[0]) for card in player_hand]
    computer_ranks = [ranks.index(card[0]) for card in computer_hand]
    player_score = sum(player_ranks)
    computer_score = sum(computer_ranks)
    if player_score > computer_score:
        return "You win!"
    elif player_score < computer_score:
        return "You lose!"
    else:
        return "It's a tie!"

player_hand = deal_hand(5)
computer_hand = deal_hand(5)

display_hand("Your", player_hand)
display_hand("Computer's", computer_hand)

print(determine_winner(player_hand, computer_hand))
```
---
#### 6(B) To write a Python code for infix to postfix conversion.
#### PROGRAM
```
def infix_to_postfix(expression):
    precedence = {'+': 1, '-': 1, '*': 2, '/': 2, '^': 3}
    stack = []
    postfix = []
    
    for char in expression:
        if char.isalnum():
            postfix.append(char)
        elif char == '(':
            stack.append(char)
        elif char == ')':
            while stack and stack[-1] != '(':
                postfix.append(stack.pop())
            stack.pop()  # Discard the '('
        else:  # Operator
            while stack and precedence.get(stack[-1], 0) >= precedence.get(char, 0):
                postfix.append(stack.pop())
            stack.append(char)

    while stack:
        postfix.append(stack.pop())

    return ''.join(postfix)

# Get input from user
infix_expression = input("Enter an infix expression: ")

# Display infix expression
print("Infix expression:", infix_expression)

# Convert infix to postfix and display the result
postfix_expression = infix_to_postfix(infix_expression)
print("Postfix expression:", postfix_expression)
```
---
#### 6(C). To write a Python PROGRAM for first come first serve scheduling PROGRAM.
#### PROGRAM
```
def fcfs(processes):
    # Initialize variables
    total_waiting_time = 0
    total_turnaround_time = 0

    # Set initial values for the first process
    current_time = 0

    # Process and print each process
    print("Process\tArrival Time\tBurst Time\tWaiting Time\tTurnaround Time")
    for process in processes:
        # Calculate waiting time
        waiting_time = max(0, current_time - process[1])

        # Calculate turnaround time
        turnaround_time = waiting_time + process[2]

        # Update total waiting and turnaround times
        total_waiting_time += waiting_time
        total_turnaround_time += turnaround_time

        # Update current time
        current_time += process[2]

        # Print process details
        print(f"{process[0]}\t{process[1]}\t\t{process[2]}\t\t{waiting_time}\t\t{turnaround_time}")

    # Calculate average waiting and turnaround times
    n = len(processes)
    avg_waiting_time = total_waiting_time / n
    avg_turnaround_time = total_turnaround_time / n

    # Print average times
    print(f"\nAverage Waiting Time: {avg_waiting_time}")
    print(f"Average Turnaround Time: {avg_turnaround_time}")

# Input process details
if __name__ == "__main__":
    n = int(input("Enter the number of processes: "))
    processes = []
    for i in range(1, n + 1):
        arrival_time = float(input(f"Enter arrival time for process {i}: "))
        burst_time = float(input(f"Enter burst time for process {i}: "))
        processes.append((i, arrival_time, burst_time))
    
    fcfs(processes)
```
---
#### 7(A) To write a Python script for implementing linear search technique.
#### PROGRAM
```
def linear_search(arr, x):
    for i in range(len(arr)):
        if arr[i] == x:
            return i  
    return -1  
arr = []
n = int(input("Enter the number of elements in the list: "))
for _ in range(n):
    element = float(input("Enter element: "))
    arr.append(element)
x = float(input("Enter the element to search for: "))
result = linear_search(arr, x)

if result != -1:
    print(f"Element found at index {result}")
else:
    print("Element not found")
```
---
#### 7(B) IMPLEMENTATION OF BINARY SEARCHING TECHNIQUE
###### Write a Python PROGRAM to search an element in a given linear list using recursion 
#### PROGRAM
```
def binary_search(arr, low, high, x):
    if high >= low:
        mid = low + (high - low) // 2
        if arr[mid] == x:
            return mid
        elif arr[mid] > x:
            return binary_search(arr, low, mid - 1, x)
        else:
            return binary_search(arr, mid + 1, high, x)
    else:
        return -1
arr = []
n = int(input("Enter the number of elements in the sorted list: "))
for _ in range(n):
    element = float(input("Enter element: "))
    arr.append(element)
arr.sort()
print("List:", arr)
x = float(input("Enter the element to search for: "))
result = binary_search(arr, 0, len(arr) - 1, x)

if result != -1:
    print(f"Element found at index {result}")
else:
    print("Element not found")
```
---
#### 7(C) To write a Python PROGRAM to arrange the given elements using bubble sort.
#### PROGRAM
```
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n-i-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
arr = list(map(float, input("Enter elements of the list separated by spaces: ").split()))
bubble_sort(arr)
print("Sorted list:", arr)
```
---
#### 7(D) To write a Python PROGRAM to arrange the given elements using insertion sort.
#### PROGRAM
```
def insertion_sort(arr):
    for i in range(1, len(arr)):
        key = arr[i]
        j = i - 1
        while j >= 0 and key < arr[j]:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key

n = int(input("Enter the number of elements: "))
arr = []
for _ in range(n):
    element = float(input("Enter element: "))
    arr.append(element)

insertion_sort(arr)

print("Sorted array is:", arr)
```
---
#### 7(E) To write a Python PROGRAM to arrange the given elements using selection sort.
#### PROGRAM
```
def selection_sort(arr):
    n = len(arr)
    for i in range(n):
        min_idx = i
        for j in range(i+1, n):
            if arr[j] < arr[min_idx]:
                min_idx = j
        arr[i], arr[min_idx] = arr[min_idx], arr[i]
    return arr

arr = input("Enter the elements separated by spaces: ").split()
arr = [float(num) for num in arr]

sorted_arr = selection_sort(arr)
print("Sorted array is:", sorted_arr)
```
---
#### 8(A) To write a Python PROGRAM to print a binary tree in vertical order.
#### PROGRAM
```
# Node structure for the binary tree
class Node:
    def __init__(self, key):
        self.key = key
        self.left = None
        self.right = None

# Function to perform a vertical traversal of the binary tree and populate the hash map
def vertical_order(root, hd, map):
    if root is None:
        return

    if hd in map:
        map[hd].append(root.key)
    else:
        map[hd] = [root.key]

    vertical_order(root.left, hd - 1, map)
    vertical_order(root.right, hd + 1, map)

# Function to print the binary tree in vertical order
def print_vertical_order(root):
    map = {}
    vertical_order(root, 0, map)

    # Sort the hash map based on keys (horizontal distance)
    sorted_map = sorted(map.items())

    for _, values in sorted_map:
        for value in values:
            print(value, end=" ")
        print()

# Function to build a binary tree from user input
def build_binary_tree():
    root_value = int(input("Enter the value of the root node: "))
    root = Node(root_value)

    nodes = [(root, "root")]
    while nodes:
        current, position = nodes.pop(0)

        left_value = input(f"Enter the value of the left child of {position} (or 'None' if no left child): ")
        if left_value.lower() != "none":
            left_node = Node(int(left_value))
            current.left = left_node
            nodes.append((left_node, "left child of " + str(current.key)))

        right_value = input(f"Enter the value of the right child of {position} (or 'None' if no right child): ")
        if right_value.lower() != "none":
            right_node = Node(int(right_value))
            current.right = right_node
            nodes.append((right_node, "right child of " + str(current.key)))

    return root

# Build the binary tree
root = build_binary_tree()

# Print the binary tree in vertical order
print("Vertical order traversal of binary tree is:")
print_vertical_order(root)
```
##
<details>
  <summary><h4>Sample Output for 8A</h4></summary>
 
 ```
Enter the value of the root node: 1
Enter the value of the left child of root (or 'None' if no left child): 2
Enter the value of the right child of root (or 'None' if no right child): 3
Enter the value of the left child of left child of 1 (or 'None' if no left child): 4
Enter the value of the right child of left child of 1 (or 'None' if no right child): 5
Enter the value of the left child of right child of 1 (or 'None' if no left child): none
Enter the value of the right child of right child of 1 (or 'None' if no right child): 6
Enter the value of the left child of left child of 2 (or 'None' if no left child): None
Enter the value of the right child of left child of 2 (or 'None' if no right child): none
Enter the value of the left child of right child of 2 (or 'None' if no left child): none
Enter the value of the right child of right child of 2 (or 'None' if no right child): None
Enter the value of the left child of right child of 3 (or 'None' if no left child): none
Enter the value of the right child of right child of 3 (or 'None' if no right child): NoNe
Vertical order traversal of binary tree is:
4
2
1 5
3
6
```
</details>

---
#### 9A. To write a Python PROGRAM in order to traverse and search element from binary tree 🌲 .
### PROGRAM

<details>
  <summary><h5>74 lines of code(Not Verified)</h5></summary>

```
class Node:
    def __init__(self, key):
        self.left = None
        self.right = None
        self.val = key
        self.index = None

def inorder_traversal(root):
    if root:
        inorder_traversal(root.left)
        print(f"{root.val} at position {root.index}", end=" ")
        inorder_traversal(root.right)

def preorder_traversal(root):
    if root:
        print(f"{root.val} at position {root.index}", end=" ")
        preorder_traversal(root.left)
        preorder_traversal(root.right)

def postorder_traversal(root):
    if root:
        postorder_traversal(root.left)
        postorder_traversal(root.right)
        print(f"{root.val} at position {root.index}", end=" ")

def search_element(root, key, index=1):
    if root is None or root.val == key:
        if root:
            root.index = index
        return root

    if root.val < key:
        return search_element(root.right, key, 2 * index + 1)

    return search_element(root.left, key, 2 * index)

# Function to insert a new node with the given key
def insert_node(root, key):
    if root is None:
        return Node(key)
    else:
        if key < root.val:
            root.left = insert_node(root.left, key)
        else:
            root.right = insert_node(root.right, key)
    return root

# Get user input to build the binary tree
def build_binary_tree():
    root = None
    elements = []
    n = int(input("Enter the number of elements in the binary tree: "))
    print("Enter the elements:",)
    for _ in range(n):
        key = float(input())
        elements.append(key)
        root = insert_node(root, key)
    print("Elements entered:", elements)
    return root
root = build_binary_tree()

print("\nInorder traversal:")
inorder_traversal(root)
print("\nPreorder traversal:")
preorder_traversal(root)
print("\nPostorder traversal:")
postorder_traversal(root)

key = float(input("\nEnter the element to search in the tree: "))
result = search_element(root, key)
if result:
    print(f"Element {key} found in the tree at position {result.index}.")
else:
    print(f"Element {key} not found in the tree.")
    
```
</details>

#### EASIEST VERSION OF CODE
```
class Node:
    def __init__(self, key):
        self.left = None
        self.right = None
        self.val = key
        self.index = None

def traverse(root, order=lambda r: (root := r) and traverse(r.left) + [r.val] + traverse(r.right) if r else []):
    return order(root)

def search(root, key, index=1):
    if not root or root.val == key: root and setattr(root, 'index', index)
    return root if not root or root.val == key else search(root.right, key, 2 * index + 1) if root.val < key else search(root.left, key, 2 * index)

def insert(root, key):
    return Node(key) if not root else (setattr(root, 'left', insert(root.left, key)) if key < root.val else setattr(root, 'right', insert(root.right, key)), root)[1]

def build_tree():
    root, elements = None, []
    for _ in range(int(input("Enter the number of elements in the binary tree: "))): elements.append(float(input())) ; root = insert(root, elements[-1])
    print("Elements entered:", elements)
    return root

root = build_tree()
print("\nInorder traversal:", traverse(root))
print("Preorder traversal:", traverse(root, lambda r: [r.val] + traverse(r.left) + traverse(r.right)))
print("Postorder traversal:", traverse(root, lambda r: traverse(r.left) + traverse(r.right) + [r.val]))

key = float(input("\nEnter the element to search in the tree: "))
result = search(root, key)
print(f"Element {key} found in the tree at position {result.index}." if result else f"Element {key} not found in the tree.")
```
---
#### 9(B). Python PROGRAM for preorder traversal to search element from binary tree🌲...
#### PROGRAM
```
class Node:
    def __init__(self, key):
        self.left = None
        self.right = None
        self.val = key

def insert(root, key):
    if root is None:
        return Node(key)
    else:
        if root.val < key:
            root.right = insert(root.right, key)
        else:
            root.left = insert(root.left, key)
    return root

def preorder_traversal(root, index=0, index_dict=None):
    if index_dict is None:
        index_dict = {}
    if root:
        index_dict[root.val] = index
        index = preorder_traversal(root.left, index + 1, index_dict)
        index = preorder_traversal(root.right, index + 1, index_dict)
    return index

# Get input from user to build the binary search tree
root_val = int(input("Enter the value for the root node: "))
root = Node(root_val)
elements = [root_val]

while True:
    val = input("Enter a value to insert into the binary search tree (or type 'done' to finish): ")
    if val.lower() == 'done':
        break
    elements.append(int(val))
    insert(root, int(val))

# Display all entered elements
print("\nEntered elements:", elements)

# Perform preorder traversal
print("\nPreorder Traversal:")
index_dict = {}
preorder_traversal(root, index_dict=index_dict)

# Get the element to search from the user
key = int(input("\nEnter an element to search for in the binary search tree: "))

if key in index_dict:
    print(f"Element {key} found at index {index_dict[key]} in the preorder traversal.")
else:
    print(f"Element {key} not found in the tree.")

```
---
#### 9(C).Python PROGRAM for postorder traversal to search element from binary tree...
#### PROGRAM
```
class Node:
    def __init__(self, key):
        self.left = None
        self.right = None
        self.val = key

def insert(root, key):
    if root is None:
        return Node(key)
    else:
        if root.val < key:
            root.right = insert(root.right, key)
        else:
            root.left = insert(root.left, key)
    return root

def postorder_traversal(root, index=0, index_dict=None):
    if index_dict is None:
        index_dict = {}
    if root:
        index = postorder_traversal(root.left, index, index_dict)
        index = postorder_traversal(root.right, index, index_dict)
        index_dict[root.val] = index
        index += 1
    return index

# Get input from user to build the binary search tree
root_val = int(input("Enter the value for the root node: "))
root = Node(root_val)
entered_values = [root_val]

while True:
    val = input("Enter a value to insert into the binary search tree (or type 'done' to finish): ")
    if val.lower() == 'done':
        break
    entered_values.append(int(val))
    insert(root, int(val))

# Display all entered values
print("\nEntered values:", entered_values)

# Perform postorder traversal
print("\nPostorder Traversal:")
index_dict = {}
postorder_traversal(root, index_dict=index_dict)

# Get the element to search from the user
key = int(input("\nEnter an element to search for in the binary search tree: "))

if key in index_dict:
    print(f"Element {key} found at index {index_dict[key]} in the postorder traversal.")
else:
    print(f"Element {key} not found in the tree.")
```
---

#### 10(A)
```
from collections import defaultdict

class Node:
    def __init__(self, key):
        self.left = None
        self.right = None
        self.val = key

def insert(root, key):
    if root is None:
        return Node(key)
    else:
        if root.val < key:
            root.right = insert(root.right, key)
        else:
            root.left = insert(root.left, key)
    return root

def vertical_order_traversal(root, hd, map):
    if root is None:
        return
    if hd in map:
        map[hd].append(root.val)
    else:
        map[hd] = [root.val]
    vertical_order_traversal(root.left, hd-1, map)
    vertical_order_traversal(root.right, hd+1, map)

def print_vertical_order(map):
    for key in sorted(map.keys()):
        for value in map[key]:
            print(value, end=" ")
        print()

# Get input from user to build the binary search tree
root_val = int(input("Enter the value for the root node: "))
root = Node(root_val)

while True:
    val = input("Enter a value to insert into the binary search tree (or type 'done' to finish): ")
    if val.lower() == 'done':
        break
    insert(root, int(val))

# Perform vertical order traversal
print("\nVertical Order Traversal:")
map = defaultdict(list)
vertical_order_traversal(root, 0, map)
print_vertical_order(map)
```
---
#### 10(B)
#### PROGRAM

```
class Node:
    def __init__(self, key):
        self.left = self.right = None
        self.val = key

def insert(root, key):
    if not root: return Node(key)
    if root.val < key: root.right = insert(root.right, key)
    else: root.left = insert(root.left, key)
    return root

def search(root, key, pos=1):
    if not root or root.val == key: return root, pos
    return search(root.right, key, pos*2+1) if root.val < key else search(root.left, key, pos*2)

root = None
for key in map(int, input("Enter elements to insert (space-separated): ").split()): root = insert(root, key)

element_to_search = int(input("Enter element to search: "))
result, position = search(root, element_to_search)

print(f"Element {element_to_search} found at position {position}." if result else f"Element {element_to_search} not found in the tree.")
```
---
#### 11(A)
#### PROGRAM
<details>
<summary>
Algorithm :
</summary>

![Alt Text]()


1. Define a function `is_min_heap(arr)` that takes an array `arr` as input:
   - Iterate over the array up to the parent nodes (`range(len(arr)//2)`):
     - For each parent node at index `i`:
       - If the left child exists (`2*i + 1 < len(arr)`) and the value at the parent node is greater than the value at the left child (`arr[i] > arr[2*i + 1]`), return `False`.
       - If the right child exists (`2*i + 2 < len(arr)`) and the value at the parent node is greater than the value at the right child (`arr[i] > arr[2*i + 2]`), return `False`.
   - If the loop completes without returning `False`, return `True`.

2. Get input from the user:
   - Prompt the user to enter array elements (space-separated).
   - Split the input string into individual elements, convert them to integers, and store them in the `arr` list.

3. Check if the array is a min heap:
   - Call the `is_min_heap` function with the `arr` list.
   - If the return value is `True`, print "The array is a min heap.".
   - If the return value is `False`, print "The array is not a min heap.".
   - 
</details>

##


```
def is_min_heap(arr):
    n = len(arr)
    positions = []
    for i in range(n // 2):
        if 2*i + 1 < n and arr[i] > arr[2*i + 1]:
            positions.append((i, 2*i + 1))
        if 2*i + 2 < n and arr[i] > arr[2*i + 2]:
            positions.append((i, 2*i + 2))
    return len(positions) == 0, positions

# Get array elements from user
arr = list(map(int, input("Enter the array elements (space-separated): ").split()))

# Check if the array is a min heap
is_heap, positions = is_min_heap(arr)
if is_heap:
    print("The array is a min heap.")
else:
    print("The array is not a min heap. Positions of elements violating min heap property:")
    for parent, child in positions:
        print(f"Parent at index {parent}, Child at index {child}")

```
---
#### 11(B)
<details>
<summary>
Algorithm :
</summary>

![Alt Text]()


1. Define a function `is_max_heap(arr)` that takes an array `arr` as input:
   - Iterate over the array up to the parent nodes (`for i in range(len(arr)//2)`):
     - If the left child exists and is greater than the parent, return `False`.
     - If the right child exists and is greater than the parent, return `False`.
   - If no violations are found, return `True`.

2. Get input from the user:
   - Prompt the user to enter array elements (space-separated).
   - Split the input string into individual elements, convert them to integers, and store them in the `arr` list.

3. Check if the array is a max heap:
   - Call the `is_max_heap` function with the `arr` list.
   - If the return value is `True`, print "The array is a max heap.".
   - If the return value is `False`, print "The array is not a max heap.".
   - 

</details>

##


#### PROGRAM
```
def is_max_heap(arr):
    n = len(arr)
    positions = []
    for i in range(n // 2):
        if 2*i + 1 < n and arr[i] < arr[2*i + 1]:
            positions.append((i, 2*i + 1))
        if 2*i + 2 < n and arr[i] < arr[2*i + 2]:
            positions.append((i, 2*i + 2))
    return len(positions) == 0, positions

# Get array elements from user
arr = list(map(int, input("Enter the array elements (space-separated): ").split()))

# Check if the array is a max heap
is_heap, positions = is_max_heap(arr)
if is_heap:
    print("The array is a max heap.")
else:
    print("The array is not a max heap. Positions of elements violating max heap property:")
    for parent, child in positions:
        print(f"Parent at index {parent}, Child at index {child}")

```
---
#### 12(A)
<details>
<summary>
Algorithm :
</summary>

![Alt Text]()



1. Define a class `Graph` with the following methods:
   - `__init__(self, num_nodes)`: Initialize the graph with `num_nodes` nodes and create an adjacency matrix with all entries set to 0.
   - `add_edge(self, u, v)`: Add an undirected edge between nodes `u` and `v` by setting the corresponding entries in the adjacency matrix to 1.
   - `display(self)`: Display the adjacency matrix of the graph.

2. Main program:
   - Get the number of nodes from the user (`num_nodes`).
   - Create an instance of the `Graph` class with `num_nodes`.
   - Continuously get edges from the user in the format 'u v' and add them to the graph until the user enters 'done'.
   - Display the adjacency matrix of the graph.


</details>

##
#### PROGRAM
```
class Graph:
    def __init__(self, num_nodes):
        self.num_nodes = num_nodes
        self.adj_matrix = [[0] * num_nodes for _ in range(num_nodes)]

    def add_edge(self, u, v):
        if u < 0 or u >= self.num_nodes or v < 0 or v >= self.num_nodes:
            raise ValueError("Invalid node index")
        self.adj_matrix[u][v] = 1
        self.adj_matrix[v][u] = 1

    def display(self):
        for row in self.adj_matrix:
            print(" ".join(map(str, row)))

# Get number of nodes from user
num_nodes = int(input("Enter the number of nodes: "))
g = Graph(num_nodes)

# Get edges from user
while True:
    edge = input("Enter an edge (format: 'u v', or type 'done' to finish): ")
    if edge.lower() == 'done':
        break
    u, v = map(int, edge.split())
    g.add_edge(u, v)

print("\nAdjacency Matrix:")
g.display()
```
---
#### 12(B)
<details>
<summary>
Algorithm :
</summary>

![Alt Text]()



1. Graph Class: Defines a class `Graph` with the following methods:
   - `__init__(self)`: Initializes the graph with an empty adjacency list.
   - `add_edge(self, u, v)`: Adds an undirected edge between nodes `u` and `v` by appending `v` to the adjacency list of `u` and vice versa.
   - `display(self)`: Displays the adjacency list of the graph.

2. Main Program:
   - Creates an instance of the `Graph` class.
   - Allows the user to add edges to the graph by entering pairs of nodes (`u` and `v`).
   - Prints the adjacency list of the graph using the `display` method.


</details>

##
#### PROGRAM
```
from collections import defaultdict

class Graph:
    def __init__(self):
        self.adj_list = defaultdict(list)

    def add_edge(self, u, v):
        self.adj_list[u].append(v)
        self.adj_list[v].append(u)

    def display(self):
        for node in self.adj_list:
            print(f"Node {node}: {self.adj_list[node]}")

# Create an instance of the Graph class
g = Graph()

# Get edges from the user
while True:
    edge = input("Enter an edge (format: 'u v', or type 'done' to finish): ")
    if edge.lower() == 'done':
        break
    u, v = map(int, edge.split())
    g.add_edge(u, v)

# Display the adjacency list
print("\nAdjacency List:")
g.display()
```
---
#### 12(C)
<details>
<summary>
Algorithm :
</summary>

![Alt Text]()



1. Graph Class: 
   - `__init__(self)`: Initializes the graph with an empty adjacency list.
   - `add_edge(self, u, v)`: Adds an undirected edge between nodes `u` and `v` by appending `v` to the adjacency list of `u` and vice versa.
   - `dfs_util(self, node, visited)`: Recursive utility function for DFS traversal. Marks the current node as visited, prints it, and recursively calls itself for all adjacent nodes that have not been visited yet.
   - `dfs(self, start)`: Initializes a visited array, calls `dfs_util` with the start node, and prints the DFS traversal starting from the start node.

2. Main Program:
   - Creates an instance of the `Graph` class.
   - Allows the user to add edges to the graph by entering pairs of nodes (`u` and `v`).
   - Gets the starting node for DFS traversal from the user.
   - Performs DFS traversal starting from the user-specified node and prints the traversal path.


</details>

##
#### PROGRAM
```
from collections import defaultdict

class Graph:
    def __init__(self):
        self.adj_list = defaultdict(list)

    def add_edge(self, u, v):
        self.adj_list[u].append(v)
        self.adj_list[v].append(u)

    def dfs_util(self, node, visited):
        visited[node] = True
        print(node, end=' ')

        for neighbor in self.adj_list[node]:
            if not visited[neighbor]:
                self.dfs_util(neighbor, visited)

    def dfs(self, start):
        visited = [False] * len(self.adj_list)
        self.dfs_util(start, visited)

# Create an instance of the Graph class
g = Graph()

# Get edges from the user
while True:
    edge = input("Enter an edge (format: 'u v', or type 'done' to finish): ")
    if edge.lower() == 'done':
        break
    u, v = map(int, edge.split())
    g.add_edge(u, v)

# Get the starting node for DFS traversal from the user
start_node = int(input("Enter the starting node for DFS traversal: "))

# Perform DFS traversal starting from the user-specified node
print("\nDFS Traversal:")
g.dfs(start_node)
```
---
#### 12(D)
<details>
<summary>
Algorithm :
</summary>

![Alt Text]()


1. `bfs(graph, start)` Function:
   - `graph` is a defaultdict with keys as vertices and values as sets of adjacent vertices.
   - `start` is the starting vertex for BFS traversal.
   - `visited` is a set to keep track of visited vertices.
   - `queue` is a deque to store vertices to be processed next.
   - `traversal` is a list to store the traversal path.
   - While the queue is not empty:
     - Pop a vertex from the left of the queue.
     - If the vertex has not been visited:
       - Mark it as visited.
       - Append it to the traversal path.
       - Add its unvisited adjacent vertices to the queue.
   - Return the traversal path.

2. Main Program:
   - Creates an empty graph as a defaultdict of sets.
   - Gets the number of edges from the user and adds edges to the graph accordingly.
   - Gets the starting index for BFS traversal from the user.
   - Calls the `bfs` function with the graph and starting index, and prints the BFS traversal path.


</details>

##
#### PROGRAM
```
from collections import defaultdict, deque
def bfs(graph, start):
    visited = set()
    queue = deque([start])
    traversal = []
    while queue:
        vertex = queue.popleft()
        if vertex not in visited:
            visited.add(vertex)
            traversal.append(vertex)
            queue.extend(graph[vertex] - visited)
    return traversal
graph = defaultdict(set)
num_edges = int(input("Enter the number of edges: "))
for _ in range(num_edges):
    edge = input("Enter edge (format: src dest): ").split()
    src, dest = map(int, edge)
    graph[src].add(dest)
    graph[dest].add(src)
start_index = int(input("Enter the starting index for BFS traversal: "))
result = bfs(graph, start_index)
print("BFS traversal:", result)
```
---
#### 13(A)
<details>
<summary>
Algorithm :
</summary>


1. **Edge Class**: Define a class `Edge` with attributes `u`, `v`, and `weight` to represent an edge.

2. **Bellman-Ford Algorithm**:
    - Initialize an array `dist` of size `V` with all elements set to infinity except for the source vertex `src`, which is set to 0.
    - Iterate `V - 1` times:
        - Iterate through all edges:
            - If `dist[u] + weight < dist[v]`, update `dist[v]` to `dist[u] + weight`.
    - Check for negative weight cycles:
        - Iterate through all edges:
            - If `dist[u] + weight < dist[v]`, print "Graph contains negative weight cycle" and return.
    - Print the vertex distance from the source.

3. **Input**:
    - Take input for the number of vertices `V`, the number of edges `E`.
    - Take input for each edge: source vertex `u`, destination vertex `v`, and weight.
    - Take input for the source vertex `src`.

4. **Output**: Print the distances from the source vertex to all other vertices.

Let me know if you need further clarification on any part!

</details>

##
#### PROGRAM
```
class Edge:
    def __init__(self, u, v, weight):
        self.u = u
        self.v = v
        self.weight = weight

def bellman_ford(edges, V, E, src):
    dist = [float("Inf")] * V
    dist[src] = 0

    for _ in range(V - 1):
        for i in range(E):
            if dist[edges[i].u] + edges[i].weight < dist[edges[i].v]:
                dist[edges[i].v] = dist[edges[i].u] + edges[i].weight

    for i in range(E):
        if dist[edges[i].u] + edges[i].weight < dist[edges[i].v]:
            print("Graph contains negative weight cycle")
            return

    print("Vertex Distance from Source")
    for i in range(V):
        print(f"{i}\t\t{dist[i]}")

V = int(input("Enter the number of vertices in the graph: "))
E = int(input("Enter the number of edges in the graph: "))
edges = []

for _ in range(E):
    u, v, weight = map(int, input("Enter the source, destination, and weight of edge (space-separated): ").split())
    edges.append(Edge(u, v, weight))

src = int(input("Enter the source vertex: "))  # Source vertex

bellman_ford(edges, V, E, src)
```
---
#### 13(B)
<details>
<summary>
Algorithm :
</summary>


1. **Edge Class**:
   - Define a class `Edge` with attributes `u`, `v`, and `weight`.

2. **Dijkstra's Algorithm**:
   - Create an empty list `graph` of size `V` to represent the graph.
   - Iterate through the edges and populate the `graph` list accordingly.
   - Initialize a priority queue `pq` with a tuple `(0, src)` representing the distance from the source vertex.
   - Initialize an array `dist` of size `V` with all elements set to infinity except for the source vertex `src`, which is set to 0.
   - While the priority queue is not empty:
       - Pop the node with the minimum cost from the priority queue.
       - Update the distance for its neighbors if a shorter path is found.
   - Print the vertex distance from the source.

3. **Input**:
   - Take input for the number of vertices `V`, the number of edges `E`.
   - Take input for each edge: source vertex `u`, destination vertex `v`, and weight.
   - Take input for the source vertex `src`.

4. **Output**: Print the distances from the source vertex to all other vertices.

</details>

##
#### PROGRAM
```
import heapq

class Edge:
    def __init__(self, u, v, weight):
        self.u = u
        self.v = v
        self.weight = weight

def dijkstra(edges, V, src):
    graph = [[] for _ in range(V)]
    for edge in edges:
        graph[edge.u].append((edge.v, edge.weight))

    pq = [(0, src)]
    dist = [float("inf")] * V
    dist[src] = 0

    while pq:
        cost, u = heapq.heappop(pq)
        if cost > dist[u]:
            continue
        for v, weight in graph[u]:
            if dist[u] + weight < dist[v]:
                dist[v] = dist[u] + weight
                heapq.heappush(pq, (dist[v], v))

    print("Vertex Distance from Source")
    for i in range(V):
        print(f"{i}\t\t{dist[i]}")

# Get input from user for the graph
V = int(input("Enter the number of vertices in the graph: "))
E = int(input("Enter the number of edges in the graph: "))
edges = []

for _ in range(E):
    u, v, weight = map(int, input("Enter the source, destination, and weight of edge (space-separated): ").split())
    edges.append(Edge(u, v, weight))

src = int(input("Enter the source vertex: "))  # Source vertex

dijkstra(edges, V, src)
```
---
#### 14(A)
<details>
<summary>
Algorithm :
</summary>

1. **Edge Class**: Defines a class `Edge` to represent an edge with attributes `u`, `v`, and `weight`.

2. **prim Function**: Takes a list of edges and the number of vertices `V` as input.
    - Creates an adjacency list representation of the graph.
    - Initializes a priority queue `pq` with tuples `(weight, vertex)` to keep track of the next vertices to explore.
    - Uses a while loop to iteratively select the next edge with the minimum weight.
    - Adds the selected edge to the MST and updates the total weight.
    - Prints the edges in the MST and the total weight.

3. **Input**: Takes input for the number of vertices `V`, the number of edges `E`, and the edges themselves.

4. **Output**: Prints the edges in the Minimum Spanning Tree and the total weight.

</details>

##
#### PROGRAM
```
import heapq

class Edge:
    def __init__(self, u, v, weight):
        self.u = u
        self.v = v
        self.weight = weight

def prim(edges, V):
    graph = [[] for _ in range(V)]
    for edge in edges:
        graph[edge.u].append((edge.v, edge.weight))
        graph[edge.v].append((edge.u, edge.weight))

    visited = [False] * V
    mst_edges = []
    total_weight = 0

    start_vertex = int(input("Enter the starting vertex (0 to V-1): "))
    pq = [(0, start_vertex)]  # Start from the specified vertex
    while pq:
        weight, u = heapq.heappop(pq)
        if visited[u]:
            continue
        visited[u] = True
        total_weight += weight
        for v, w in graph[u]:
            if not visited[v]:
                heapq.heappush(pq, (w, v))
                mst_edges.append(Edge(u, v, w))

    print("\nEdges in the Minimum Spanning Tree:")
    for edge in mst_edges:
        print(f"{edge.u} - {edge.v}  Weight: {edge.weight}")
    print("Total Weight of the Minimum Spanning Tree:", total_weight)

# Get input from user for the graph
V = int(input("Enter the number of vertices in the graph: "))
E = int(input("Enter the number of edges in the graph: "))
edges = []

for _ in range(E):
    u, v, weight = map(int, input("Enter the source, destination, and weight of edge (space-separated): ").split())
    edges.append(Edge(u, v, weight))

prim(edges, V)
```
---
#### 14(B)
<details>
<summary>
Algorithm :
</summary>

1. Define an `Edge` class to represent edges.
2. Implement `find` to find the parent of a node in a disjoint set.
3. Implement `union` to merge two sets in a disjoint set.
4. Implement `kruskal` to find the Minimum Spanning Tree using Kruskal's algorithm.
5. Sort edges by weight.
6. Initialize MST, parent, and rank arrays.
7. Iterate through sorted edges.
    - Find the roots of sets containing edge endpoints.
    - If roots are different, add edge to MST and merge sets.
8. Print MST edges and total weight.

</details>

##
#### PROGRAM:
```
class Edge:
    def __init__(self, u, v, weight):
        self.u = u
        self.v = v
        self.weight = weight

def find(parent, i):
    if parent[i] == i:
        return i
    return find(parent, parent[i])

def union(parent, rank, x, y):
    x_root = find(parent, x)
    y_root = find(parent, y)

    if rank[x_root] < rank[y_root]:
        parent[x_root] = y_root
    elif rank[x_root] > rank[y_root]:
        parent[y_root] = x_root
    else:
        parent[y_root] = x_root
        rank[x_root] += 1

def kruskal(edges, V):
    edges.sort(key=lambda edge: edge.weight)
    mst = []
    parent = [i for i in range(V)]
    rank = [0] * V
    total_weight = 0

    for edge in edges:
        u = edge.u
        v = edge.v
        u_root = find(parent, u)
        v_root = find(parent, v)
        if u_root != v_root:
            mst.append(edge)
            total_weight += edge.weight
            union(parent, rank, u_root, v_root)

    print("Edges in the Minimum Spanning Tree:")
    for edge in mst:
        print(f"{edge.u} - {edge.v}  Weight: {edge.weight}")
    print("Total Weight of the Minimum Spanning Tree:", total_weight)

# Get input from user for the graph
V = int(input("Enter the number of vertices in the graph: "))
E = int(input("Enter the number of edges in the graph: "))
edges = []

for _ in range(E):
    u, v, weight = map(int, input("Enter the source, destination, and weight of edge (space-separated): ").split())
    edges.append(Edge(u, v, weight))

kruskal(edges, V)
```
---
## CONTENT BEYOND SYLLABUS 
---
#### Heap Sort
<details>
<summary>
Topic and Aim :
</summary>

 ##### Topic : Heap Sort Implementation
##### Aim : To implement the Heap Sort algorithm in Python to sort an array of integers.
</details>

##

<details>
<summary>
Algorithm :
</summary>

1. **heapify(arr, n, i) Function:**
   - Set `largest` to `i`.
   - Calculate the left child index `l = 2 * i + 1` and right child index `r = 2 * i + 2`.
   - If the left child exists and is greater than the current largest element, update `largest` to `l`.
   - If the right child exists and is greater than the current largest element, update `largest` to `r`.
   - If `largest` is not equal to `i`, swap `arr[i]` and `arr[largest]`, then recursively call `heapify` on the affected child.
   
2. **heap_sort(arr) Function:**
   - Get the length of the array `n`.
   - Build a max heap by calling `heapify` on each non-leaf node, starting from the last non-leaf node down to the root.
   - Extract elements from the heap one by one:
     - Swap the first element with the last element in the heap.
     - Reduce the heap size by one.
     - Call `heapify` on the root to restore the heap property.
   - Return the sorted array.

3. **Main Program:**
   - Take input for the array elements.
   - Call `heap_sort` function with the input array.
   - Print the sorted array.

</details>

##

#### Program 
```
def heapify(arr, n, i):
    largest = i
    l = 2 * i + 1
    r = 2 * i + 2
    if l < n and arr[i] < arr[l]:
        largest = l
    if r < n and arr[largest] < arr[r]:
        largest = r
    if largest != i:
        arr[i], arr[largest] = arr[largest], arr[i]
        heapify(arr, n, largest)

def heap_sort(arr):
    n = len(arr)
    for i in range(n // 2 - 1, -1, -1):
        heapify(arr, n, i)
    for i in range(n - 1, 0, -1):
        arr[i], arr[0] = arr[0], arr[i]
        heapify(arr, i, 0)
    return arr

arr = [int(x) for x in input("Enter the array elements separated by space: ").split()]

result = heap_sort(arr)
print("Sorted array is:", result)
```
---
#### Quick Sort 
<details>
<summary>
Topic and Aim :
</summary>

#### Topic : Quick Sort Implementation

#### Aim : To implement the Quick Sort algorithm in Python to sort an array of integers.

</details>

##
<details>
<summary>
Algorithm:
</summary>


1. **partition(arr, low, high) Function:**
   - Choose the last element `arr[high]` as the pivot.
   - Initialize the index of the smaller element `i` to `low - 1`.
   - Iterate through the array from index `low` to `high - 1`.
     - If the current element `arr[j]` is less than the pivot:
       - Increment `i`.
       - Swap `arr[i]` and `arr[j]`.
   - Swap `arr[i + 1]` and the pivot `arr[high]`.
   - Return the index `i + 1`, which is the partitioning index.

2. **quick_sort(arr, low, high) Function:**
   - If `low` is less than `high`:
     - Call `partition` to get the partitioning index `pi`.
     - Recursively call `quick_sort` on the sub-array before `pi`.
     - Recursively call `quick_sort` on the sub-array after `pi`.

3. **Main Program:**
   - Take input for the array elements.
   - Get the length of the array `n`.
   - Call `quick_sort` with the input array, starting from index 0 to `n - 1`.
   - Print the sorted array.


</details>

##
#### PROGRAM 
```
def partition(arr, low, high):
    pivot = arr[high]
    i = low - 1
    for j in range(low, high):
        if arr[j] < pivot:
            i += 1
            arr[i], arr[j] = arr[j], arr[i]
    arr[i + 1], arr[high] = arr[high], arr[i + 1]
    return i + 1

def quick_sort(arr, low, high):
    if low < high:
        pi = partition(arr, low, high)
        quick_sort(arr, low, pi - 1)
        quick_sort(arr, pi + 1, high)

# Take input from the user
arr = [int(x) for x in input("Enter the array elements separated by space: ").split()]

n = len(arr)
quick_sort(arr, 0, n - 1)
print("Sorted array is:", arr)
```
---
#### Merge Sort
<details>
<summary>
Topic and Aim  :
</summary>

#### Topic : Merge Sort Implementation

#### Aim : To implement the Merge Sort algorithm in Python to sort an array of integers.

</details>

##
<details>
<summary>
Algorithm :
</summary>



1. **merge_sort(arr) Function:**
   - If the array `arr` has one or fewer elements, return `arr`.
   - Calculate the middle index `mid` of the array.
   - Recursively call `merge_sort` on the left half of the array (`arr[:mid]`) and the right half of the array (`arr[mid:]`).
   - Merge the sorted left and right halves using the `merge` function.

2. **merge(left, right) Function:**
   - Initialize an empty list `result`.
   - Initialize indices `i` and `j` for the left and right arrays, respectively, both starting at 0.
   - Iterate while `i` and `j` are within their respective array bounds:
     - Append the smaller element from `left[i]` and `right[j]` to `result`.
     - Increment the index of the array from which the element was taken.
   - Extend `result` with any remaining elements in `left` or `right`.
   - Return `result`.

3. **Main Program:**
   - Take input for the array elements.
   - Call `merge_sort` with the input array.
   - Print the sorted array.



</details>

##
#### PROGRAM 
```
def merge_sort(arr):
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    return merge(left, right)

def merge(left, right):
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] < right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    result.extend(left[i:])
    result.extend(right[j:])
    return result

# Take input from the user
arr = [int(x) for x in input("Enter the array elements separated by space: ").split()]

result = merge_sort(arr)
print("Sorted array is:", result)
```
---
#### Analysis of Sorting Algorithms
##### comparing the average-case time complexity of bubble sort, selection sort, merge sort, and quick sort:
##
##
<details>
<summary>
Topic and Aim :
</summary>

#### Topic : Performance Comparison of Sorting Algorithms

#### Aim : To compare the performance of different sorting algorithms (Bubble Sort, Selection Sort, Merge Sort, and Quick Sort) in Python by measuring their execution times on a given array of integers

</details>

##
<details>
<summary>
Algorithm:
</summary>


1. **Bubble Sort Algorithm:**
   - Start with the first element and compare it with the next element.
   - If the current element is greater than the next element, swap them.
   - Repeat this process for each pair of adjacent elements until the list is sorted.
   - Time complexity: O(n^2)

2. **Selection Sort Algorithm:**
   - Find the minimum element from the unsorted part and swap it with the first element.
   - Repeat this process for the remaining unsorted part.
   - Time complexity: O(n^2)

3. **Merge Sort Algorithm:**
   - Divide the array into two halves.
   - Recursively sort each half.
   - Merge the sorted halves to produce the final sorted array.
   - Time complexity: O(n log n)

4. **Quick Sort Algorithm:**
   - Choose a pivot element from the array.
   - Partition the array into two sub-arrays: elements less than the pivot and elements greater than the pivot.
   - Recursively apply quicksort on the two sub-arrays.
   - Time complexity: O(n log n) (average case), O(n^2) (worst case)

5. **Input:**
   - Get the number of elements `n` from the user.
   - Prompt the user to enter `n` elements.
   - Store the elements in the list `arr`.

6. **Measure Sorting Time:**
   - Measure the time taken for each sorting algorithm using `time.time()` function.
   - Print the time taken for each sorting algorithm.


</details>

##
#### PROGRAM 
```
import time
import random

def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n-i-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]

def selection_sort(arr):
    n = len(arr)
    for i in range(n):
        min_idx = i
        for j in range(i+1, n):
            if arr[j] < arr[min_idx]:
                min_idx = j
        arr[i], arr[min_idx] = arr[min_idx], arr[i]

def merge_sort(arr):
    if len(arr) > 1:
        mid = len(arr) // 2
        L = arr[:mid]
        R = arr[mid:]

        merge_sort(L)
        merge_sort(R)

        i = j = k = 0

        while i < len(L) and j < len(R):
            if L[i] < R[j]:
                arr[k] = L[i]
                i += 1
            else:
                arr[k] = R[j]
                j += 1
            k += 1

        while i < len(L):
            arr[k] = L[i]
            i += 1
            k += 1

        while j < len(R):
            arr[k] = R[j]
            j += 1
            k += 1

def quick_sort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quick_sort(left) + middle + quick_sort(right)

# Get input from user
n = int(input("Enter the number of elements: "))
arr = [int(input(f"Enter element {i+1}: ")) for i in range(n)]

# Measure the time taken for each sorting algorithm
start_time = time.time()
bubble_sort(arr.copy())
print("Bubble Sort Time:", time.time() - start_time)

start_time = time.time()
selection_sort(arr.copy())
print("Selection Sort Time:", time.time() - start_time)

start_time = time.time()
merge_sort(arr.copy())
print("Merge Sort Time:", time.time() - start_time)

start_time = time.time()
quick_sort(arr.copy())
print("Quick Sort Time:", time.time() - start_time)
```
---
#### Multiway Search Trees.
<details>
<summary>
Topic , Aim and Algorithm :
</summary>

#### Topic : Multiway Tree Implementation
##
#### Aim : To implement a Multiway Tree in Python and demonstrate its insertion and search functionalities.
##
#### Algorithm :

1. **MultiwayTreeNode Class:**
   - Define a class `MultiwayTreeNode` with attributes `key` and `child`.

2. **MultiwayTree Class:**
   - Define a class `MultiwayTree` with a `root` attribute initialized to `None`.

3. **Insertion Method (`insert`):**
   - If tree is empty, create root node with given key.
   - Otherwise, recursively insert key using `_insert` method.

4. **Private Insertion Method (`_insert`):**
   - If node's key is `None`, set node's key to given key.
   - Recursively insert key based on comparison with node's key.

5. **Search Method (`search`):**
   - Call private method `_search` with root node and search key.

6. **Private Search Method (`_search`):**
   - If node is `None`, return `False`.
   - If node's key matches search key, return `True`.
   - Recursively search based on comparison with node's key.

7. **Main Program:**
   - Create `MultiwayTree` instance.
   - Get number of elements `n` from user.
   - Iterate `n` times to input keys and insert into tree.
   - Get search key from user and print result of search operation.


</details>

##
#### PROGRAM
```
class MultiwayTreeNode:
    def __init__(self, key=None):
        self.key = key
        self.child = []

class MultiwayTree:
    def __init__(self):
        self.root = NoneMultiway Tree Implementation

    def insert(self, key):
        if self.root is None:
            self.root = MultiwayTreeNode(key)
        else:
            self._insert(self.root, key)

    def _insert(self, node, key):
        if node.key is None:
            node.key = key
        elif key < node.key:
            if len(node.child) == 0:
                node.child.append(MultiwayTreeNode(key))
            else:
                self._insert(node.child[0], key)
        else:
            self._insert(node.child[-1], key)

    def search(self, key):
        return self._search(self.root, key)

    def _search(self, node, key):
        if node is None:
            return False
        if node.key == key:
            return True
        elif key < node.key:
            return self._search(node.child[0], key)
        else:
            return self._search(node.child[-1], key)

tree = MultiwayTree()
n = int(input("Enter the number of elements: "))
for _ in range(n):
    key = int(input("Enter key: "))
    tree.insert(key)

search_key = int(input("Enter key to search: "))
print(tree.search(search_key))
```
---
#### Radix Sort
<details>
<summary>
Topic , Aim and Algorithm :
</summary>

#### Topic : Radix Sort Implementation
##
#### Aim : To implement the Radix Sort algorithm in Python to sort an array of integers.
##
#### Algorithm :


1. **Radix Sort Function (`radix_sort`):**
   - Find the maximum number in the array.
   - Initialize `exp` to 1.
   - Iterate while `max_num // exp` is greater than 0:
     - Call `counting_sort` function with the array and `exp`.
     - Update `exp` by multiplying it by 10.

2. **Counting Sort Function (`counting_sort`):**
   - Initialize `output` and `count` arrays.
   - Iterate through the array and count the occurrences of each digit using `exp`.
   - Modify the count array to represent cumulative counts.
   - Iterate through the array in reverse:
     - Determine the index of each element based on the digit at `exp`.
     - Place the element in the correct position in the output array.
     - Decrement the count of the digit.
   - Copy the sorted output array back to the original array.

3. **Main Program:**
   - Get the number of elements `n` from the user.
   - Prompt the user to input `n` elements.
   - Call `radix_sort` function to sort the array.
   - Print the sorted array.


</details>

##
#### PROGRAM:
```
def radix_sort(arr):
    max_num = max(arr)
    exp = 1
    while max_num // exp > 0:
        counting_sort(arr, exp)
        exp *= 10

def counting_sort(arr, exp):
    n = len(arr)
    output = [0] * n
    count = [0] * 10

    for i in range(n):
        index = arr[i] // exp
        count[index % 10] += 1

    for i in range(1, 10):
        count[i] += count[i - 1]

    i = n - 1
    while i >= 0:
        index = arr[i] // exp
        output[count[index % 10] - 1] = arr[i]
        count[index % 10] -= 1
        i -= 1

    for i in range(n):
        arr[i] = output[i]

n = int(input("Enter the number of elements: "))
arr = [int(input(f"Enter element {i+1}: ")) for i in range(n)]

radix_sort(arr)
print("Sorted array is:", arr)
```
---
##### The biggest risk is not taking any risk. In a world that is changing quickly, the only strategy that is guaranteed to fail is not taking risks. – Mark Zuckerberg

---

##

### I shall expeditiously conclude the outstanding updates.
##
# GM 




