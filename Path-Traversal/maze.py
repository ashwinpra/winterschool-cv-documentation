from asyncio.windows_events import NULL
import sys

class Node:
    def __init__(self, state, parent, action):
        self.state = state
        self.parent = parent
        self.action = action
    

class StackFrontier:
    def __init__(self):
        self.frontier = []
    
    def add(self, node:Node):
        self.frontier.append(node)
    
    def contains_state(self, state):
        return any(node.state == state for node in self.frontier )
    
    def empty(self):
        return len(self.frontier) == 0

    def remove(self):
        if self.empty():
            raise Exception("empty frontier")
        else:
            node = self.frontier[-1]
            self.frontier = self.frontier[:-1]
            return node


class QueueFrontier(StackFrontier):

    def remove(self):
        if self.empty():
            raise Exception("empty frontier")
        else:
            node = self.frontier[0]
            self.frontier = self.frontier[1:]
            return node

class Maze:
    def __init__(self, filename) -> None:

        # Read the file and set the size of the maze
        with open(filename) as f:
            contents = f.read()
        
        # Validate start and goal
        if contents.count("A") !=1:
            raise Exception("maze must have exactly one start point")
        if contents.count("B") !=1:
            raise Exception("maze must have exactly one end point")
        
        # Determine the height and width of maze
        contents = contents.splitlines()
        self.height = len(contents)
        self.width = max(len(line) for line in contents)

        # Keep tract of walls
        self.walls = []
        for i in range(self.height):
            row = []
            for j in range(self.width):
                try:
                    if contents[i][j]=="A":
                        self.start = (i, j)
                        row.append(False)
                    elif contents[i][j]=="B":
                        self.goal = (i, j)
                        row.append(False)
                    elif contents[i][j]==" ":
                        row.append(False)
                    else:
                        row.append(True)
                except IndexError:
                    row.append(False)
            self.walls.append(row)
        
        self.solution = None
    
    def print(self):
        solution = self.solution[1] if self.solution  is not None else None
        print()
        for i, row in enumerate(self.walls):
            for j, col in enumerate(row):
                if col:
                    print("#", end="")
                elif (i, j) == self.start:
                    print("A", end="")
                elif (i, j) == self.goal:
                    print("B", end="")
                elif solution is not None and (i,j) in solution:
                    print("*", end="")
                else:
                    print(" ",end="")
            print()
        print()

    def neighbours(self, state):
        row, col = state

        # All possible actions
        candidates = [
            ("up", (row-1, col)),
            ("down", (row+1, col)),
            ("left", (row, col - 1)),
            ("right", (row, col + 1))
        ]
        
        # Ensure actions are valid
        result = [] 
        for action, (r, c) in candidates:
            try:
                if not self.walls[r][c]:
                    result.append((action, (r, c)))
            except IndexError:
                continue
        return result
    
    def solve(self):
        '''Find solution to the matrix, if one exists'''

        # Keep Track of number of states explored
        self.number_explored = 0

        # Initialize frontier to just the starting position
        start = Node(state=self.start, parent=None, action=None)
        # frontier = StackFrontier() # For DFS
        frontier = QueueFrontier() # For BFS
        frontier.add(start)

        # Initialize an empty explored set
        self.explored = set()

        # Keep looking util the end is found
        while True:

            # If nothing left in the frontier, no path
            if frontier.empty():
                raise Exception('No solution')
            
            # Choose a node from the frontier
            node = frontier.remove()
            self.number_explored +=1

            # If node is the goal, then we have a solution
            if node.state == self.goal:
                actions = []
                cells = []

                # Follow the parent nodes to find solution
                while node.parent is not None:
                    actions.append(node.action)
                    cells.append(node.state)
                    node = node.parent
                actions.reverse()
                cells.reverse()
                self.solution = (actions, cells)
                return

            # Mark node as explored
            self.explored.add(node.state)

            # Add neighbours to the frontier
            for action, state in self.neighbours(node.state):
                if not frontier.contains_state(state) and state not in self.explored:
                    child = Node(state, node,action)
                    frontier.add(child)
    

    def output_image(self, filename, show_solution=True, show_explored=False):
        from PIL import Image, ImageDraw
        cell_size = 50
        cell_border = 2

        # create a blank canvas
        img = Image.new(
            "RGBA",
            (self.width*cell_size, self.height*cell_size),
            "black"
        )
        draw = ImageDraw.Draw(img)

        solution = self.solution[1] if self.solution is not None else None
        for i, row in enumerate(self.walls):
            for j, col in enumerate(row):

                # Walls
                if col:
                    fill = (40, 40, 40)

                # Start
                elif (i, j) == self.start:
                    fill = (255, 0, 0)

                # Goal
                elif (i, j) == self.goal:
                    fill = (0, 171, 28)

                # Solution
                elif solution is not None and show_solution and (i, j) in solution:
                    fill = (220, 235, 113)

                # Explored
                elif solution is not None and show_explored and (i, j) in self.explored:
                    fill = (212, 97, 85)

                # Empty cell
                else:
                    fill = (237, 240, 252)

                # Draw cell
                draw.rectangle(
                    ([(j * cell_size + cell_border, i * cell_size + cell_border),
                      ((j + 1) * cell_size - cell_border, (i + 1) * cell_size - cell_border)]),
                    fill=fill
                )

        img.save(filename)


if len(sys.argv) != 2:
    sys.exit("Usage: python maze.py maze.txt")

m = Maze(sys.argv[1])
print("Maze:")
m.print()
print("Solving...")
m.solve()
print("States Explored:", m.number_explored)
print("Solution:")
m.print()
m.output_image("maze.png", show_explored=True)
