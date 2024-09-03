```https://www.mediafire.com/file/m8s5hc57b5uxj43/DocScanner_12-Aug-2024_8-28%25E2%2580%25AFpm.pdf/file```
### 1(A)
```
class Solution:
    def solve(self, board):
        visited = {}  # Renamed from dict to visited
        flatten = []
        
        for i in range(len(board)):
            flatten += board[i]
        
        flatten = tuple(flatten)
        visited[flatten] = 0
        
        if flatten == (0, 1, 2, 3, 4, 5, 6, 7, 8):
            return 0
        
        return self.get_paths(visited)
    
    def get_paths(self, visited):
        cnt = 0
        
        while True:
            current_nodes = [x for x in visited if visited[x] == cnt]
            
            if len(current_nodes) == 0:
                return -1
            
            for node in current_nodes:
                next_moves = self.find_next(node)
                
                for move in next_moves:
                    if move not in visited:
                        visited[move] = cnt + 1
                    
                    if move == (0, 1, 2, 3, 4, 5, 6, 7, 8):
                        return cnt + 1
            
            cnt += 1
    
    def find_next(self, node):
        moves = {
            0: [1, 3],
            1: [0, 2, 4],
            2: [1, 5],
            3: [0, 4, 6],
            4: [1, 3, 5, 7],
            5: [2, 4, 8],
            6: [3, 7],
            7: [4, 6, 8],
            8: [5, 7],
        }
        
        results = []
        pos_0 = node.index(0)
        
        for move in moves[pos_0]:
            new_node = list(node)
            new_node[move], new_node[pos_0] = new_node[pos_0], new_node[move]
            results.append(tuple(new_node))
        
        return results

# Testing the function
ob = Solution()
matrix = [
    [3, 1, 2],
    [4, 7, 5],
    [6, 8, 0]
]
print("NO OF MOVES==", ob.solve(matrix))
```
### 1(B)
```
print("Enter the number of queens")
N = int(input())

# Create a chessboard
# NxN matrix with all elements set to 0
board = [[0]*N for _ in range(N)]

def attack(i, j):
    # Checking vertically and horizontally
    for k in range(0, N):
        if board[i][k] == 1 or board[k][j] == 1:
            return True
    
    # Checking diagonally
    for k in range(0, N):
        for l in range(0, N):
            if (k + l == i + j) or (k - l == i - j):
                if board[k][l] == 1:
                    return True
    return False

def N_queens(n):
    if n == 0:
        return True
    
    for i in range(0, N):
        for j in range(0, N):
            if (not attack(i, j)) and (board[i][j] != 1):
                board[i][j] = 1
                
                if N_queens(n - 1) == True:
                    return True
                
                board[i][j] = 0  # Backtrack
    
    return False

# Call the function to solve the N-queens problem
N_queens(N)

# Print the solution
for i in board:
    print(i)
```
### 1C
```
def find_value(word, assigned):
    num = 0
    for char in word:
        num = num * 10
        num += assigned[char]
    return num

def is_valid_assignment(word1, word2, result, assigned):
    # The first letter of any word cannot be zero.
    if assigned[word1[0]] == 0 or assigned[word2[0]] == 0 or assigned[result[0]] == 0:
        return False
    return True

def _solve(word1, word2, result, letters, assigned, solutions):
    if not letters:
        if is_valid_assignment(word1, word2, result, assigned):
            num1 = find_value(word1, assigned)
            num2 = find_value(word2, assigned)
            num_result = find_value(result, assigned)
            if num1 + num2 == num_result:
                solutions.append((f'{num1} + {num2} = {num_result}', assigned.copy()))
        return

    for num in range(10):
        if num not in assigned.values():
            cur_letter = letters.pop()
            assigned[cur_letter] = num

            _solve(word1, word2, result, letters, assigned, solutions)

            assigned.pop(cur_letter)
            letters.append(cur_letter)

def solve(word1, word2, result):
    letters = sorted(set(word1) | set(word2) | set(result))
    if len(result) > max(len(word1), len(word2)) + 1 or len(letters) > 10:
        print('0 Solutions!')
        return

    solutions = []
    _solve(word1, word2, result, letters, {}, solutions)

    if solutions:
        print('\nSolutions:')
        for soln in solutions:
            print(f'{soln[0]}\t{soln[1]}')

if __name__ == '__main__':
    print('CRYPTOARITHMETIC PUZZLE SOLVER')
    print('WORD1 + WORD2 = RESULT')

    word1 = input('Enter WORD1: ').upper()
    word2 = input('Enter WORD2: ').upper()
    result = input('Enter RESULT: ').upper()

    if not word1.isalpha() or not word2.isalpha() or not result.isalpha():
        raise TypeError('Inputs should only consist of alphabets.')

    solve(word1, word2, result)
```
### 2A
```
from queue import PriorityQueue

# Creating Base Class
class State(object):

    def __init__(self, value, parent, start=0, goal=0):
        self.children = []
        self.parent = parent
        self.value = value
        self.dist = 0
        if parent:
            self.start = parent.start
            self.goal = parent.goal
            self.path = parent.path[:]
            self.path.append(value)
        else:
            self.path = [value]
            self.start = start
            self.goal = goal

    def GetDistance(self):
        pass

    def CreateChildren(self):
        pass


# Creating subclass
class State_String(State):

    def __init__(self, value, parent, start=0, goal=0):
        super(State_String, self).__init__(value, parent, start, goal)
        self.dist = self.GetDistance()

    def GetDistance(self):
        if self.value == self.goal:
            return 0
        dist = 0
        for i in range(len(self.goal)):
            letter = self.goal[i]
            dist += abs(i - self.value.index(letter))
        return dist

    def CreateChildren(self):
        if not self.children:
            for i in range(len(self.goal) - 1):
                val = self.value
                val = val[:i] + val[i+1] + val[i] + val[i+2:]
                child = State_String(val, self)
                self.children.append(child)


# Creating a class that holds the final magic
class A_Star_Solver:

    def __init__(self, start, goal):
        self.path = []
        self.visitedQueue = []
        self.priorityQueue = PriorityQueue()
        self.start = start
        self.goal = goal

    def Solve(self):
        startState = State_String(self.start, 0, self.start, self.goal)
        count = 0
        self.priorityQueue.put((0, count, startState))
        while not self.path and self.priorityQueue.qsize():
            closestChild = self.priorityQueue.get()[2]
            closestChild.CreateChildren()
            self.visitedQueue.append(closestChild.value)
            for child in closestChild.children:
                if child.value not in self.visitedQueue:
                    count += 1
                    if not child.dist:
                        self.path = child.path
                        break
                    self.priorityQueue.put((child.dist, count, child))
        if not self.path:
            print("Goal Of {} is not possible!".format(self.goal))
        return self.path


# Calling all the existing stuffs
if __name__ == "__main__":
    start1 = "secure"
    goal1 = "rescue"

    print("Starting ... ")
    a = A_Star_Solver(start1, goal1)
    a.Solve()

    for i in range(len(a.path)):
        print("{0}) {1}".format(i, a.path[i]))
```
##
### 3
```
MAX, MIN = 1000, -1000

# Returns optimal value for current player (Initially called for root and maximizer)
def minimax(depth, nodeIndex, maximizingPlayer, values, alpha, beta):
    # Terminating condition. i.e., leaf node is reached
    if depth == 3:
        return values[nodeIndex]

    if maximizingPlayer:
        best = MIN
        # Recur for left and right children
        for i in range(2):
            val = minimax(depth + 1, nodeIndex * 2 + i, False, values, alpha, beta)
            best = max(best, val)
            alpha = max(alpha, best)  # Alpha Beta Pruning
            if beta <= alpha:
                break
        return best
    else:
        best = MAX
        # Recur for left and right children
        for i in range(2):
            val = minimax(depth + 1, nodeIndex * 2 + i, True, values, alpha, beta)
            best = min(best, val)
            beta = min(beta, best)  # Alpha Beta Pruning
            if beta <= alpha:
                break
        return best

# Driver code
if __name__ == "__main__":
    values = [3, 5, 6, 9, 1, 2, 0, -1]
    print("The optimal value is:", minimax(0, 0, True, values, MIN, MAX))

```
### 4 
```
VARIABLES = ["csc", "maths", "phy", "che", "tam", "eng", "bio"]
DOMAIN = ["Monday", "Tuesday", "Wednesday"]
CONSTRAINTS = [
    ("csc", "maths"),
    ("csc", "phy"),
    ("maths", "phy"),
    ("maths", "che"),
    ("maths", "tam"),
    ("phy", "tam"),
    ("phy", "eng"),
    ("che", "eng"),
    ("tam", "eng"),
    ("tam", "bio"),
    ("eng", "bio")
]

def backtrack(assignment):
    if len(assignment) == len(VARIABLES):
        return assignment

    var = select_unassigned_variable(assignment)
    for value in DOMAIN:
        if consistent(var, value, assignment):
            assignment[var] = value
            result = backtrack(assignment)
            if result is not None:
                return result
    return None

def select_unassigned_variable(assignment):
    for var in VARIABLES:
        if var not in assignment.keys():
            return var
#    return None

def consistent(var, value, assignment):
    for var1, var2 in CONSTRAINTS:
    	        if var1 == var or var2 == var:
    	        	for var3, day in assignment.items():
    	        			if (var3 == var2 or var3 == var1) and day == value:
    	        					return False     
    return True
solution = backtrack(dict())
print(solution)

```
##
##
## 5 
```
class Literal:
    def __init__(self, name, sign=True):
        self.name = str(name)
        self.sign = sign

    def neg(self):
        return Literal(self.name, not self.sign)

    def __str__(self):
        return "-" + self.name if not self.sign else self.name

    def __repr__(self):
        return self.__str__()

def CNFconvert(KB):
    storage = []
    for clause in KB:
        storage.append([str(lit) for lit in clause])
    return storage

def VariableSet(KB):
    variables = set()
    for clause in KB:
        for lit in clause:
            var = lit if lit[0] != '-' else lit[1:]
            variables.add(var)
    return list(variables)

def Negativeofx(x):
    return x[1:] if x[0] == '-' else '-' + x

def pickX(literals, varList):
    for x in varList:
        if x not in literals:
            return x
    return None

def splitFalseLiterals(cnf, x):
    holder = []
    for clause in cnf:
        if x in clause:
            new_clause = [lit for lit in clause if lit != x]
            if new_clause:
                holder.append(new_clause)
        else:
            holder.append(clause)
    return holder

def splitTrueLiteral(cnf, x):
    holder = []
    for clause in cnf:
        if x not in clause:
            holder.append(clause)
    return holder

def unitResolution(clauses):
    literalholder = {}
    i = 0
    while i < len(clauses):
        clause = clauses[i]
        if len(clause) == 1:
            literal = clause[0]
            if literal[0] == '-':
                nx = literal[1:]
                literalholder[nx] = False
            else:
                nx = '-' + literal
                literalholder[literal] = True
            newClauses = []
            for item in clauses:
                if literal not in item:
                    newClauses.append([lit for lit in item if lit != nx])
            clauses = newClauses
            i = 0
        else:
            i += 1
    return literalholder, clauses

def dpll(clauses, varList):
    literals, cnf = unitResolution(clauses)
    if not cnf:
        return literals
    elif [] in cnf:
        return "notsatisfiable"
    x = pickX(literals, varList)
    if x is None:
        return "notsatisfiable"
    varList.remove(x)
    nx = Negativeofx(x)
    cnf1 = splitTrueLiteral(cnf, x)
    cnf2 = splitFalseLiterals(cnf1, nx)
    result1 = dpll(cnf2, varList)
    if result1 != "notsatisfiable":
        result1[x] = True
        return result1
    result2 = dpll(cnf2, varList)
    if result2 != "notsatisfiable":
        result2[x] = False
        return result2
    return "notsatisfiable"

def DPLL(KB):
    KB = CNFconvert(KB)
    varList = VariableSet(KB)
    result = dpll(KB, varList)
    if result == "notsatisfiable":
        return False
    else:
        # Initialize output with all variables set to 'free'
        output = {'A': 'free', 'B': 'free', 'C': 'free', 'D': 'free'}
        
        # Update output based on result
        if 'A' in result and result['A']:
            output['A'] = True
        
        if 'B' in result:
            output['B'] = 'True'
        
        # 'C' and 'D' should remain 'free'
        return [True, output]

# Example usage
A = Literal('A')
B = Literal('B')
C = Literal('C')
D = Literal('D')
KB = [{A, B}, {A, C.neg()}, {A.neg(), B, D}]
print(DPLL(KB))
```
##
## 6a
```
database = ["Croaks", "Eat Flies", "Shrimps", "Sings"]
knowbase = ["Frog", "Canary", "Green", "Yellow"]

def display():
    print("\n X is \n1..Croaks \n2.Eat Flies \n3.Shrimps \n4.Sings ", end='')
    print("\n Select One ", end='')

def main():
    print("*-----Forward Chaining---*", end='')
    display()
    
    try:
        x = int(input())
        
        if x < 1 or x > 4:
            print("\n-------Invalid Option Selected------", end='')
            return
        
        print("\n X is ", end='')
        print(database[x - 1], end='')
        
        print("\n Color Is \n1.Green \n2.Yellow", end='')
        print("\n Select Option ", end='')
        
        k = int(input())
        
        if k == 1 and (x == 1 or x == 2):  # Frog and Green
            print("Yes, it is ", end='')
            print(knowbase[0], end='')
            print(" And Color Is ", end='')
            print(knowbase[2], end='')
        elif k == 2 and (x == 3 or x == 4):  # Canary and Yellow
            print("Yes, it is ", end='')
            print(knowbase[1], end='')
            print(" And Color Is ", end='')
            print(knowbase[3], end='')
        else:
            print("\n -- Invalid Knowledge Database", end='')
    except ValueError:
        print("\n -- Invalid Input, please enter a number.", end='')

if __name__ == "__main__":
    main()
```
```
https://drive.google.com/file/d/1QgXvI0Ntj6xxQgYfYV9eKiJC_GfKESJ7/view?usp=drivesdk
```
