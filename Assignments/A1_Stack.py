class node():
    def __init__(self, data):
        self.data = data

#LIFO
class Stack():
    def __init__(self, top=-1, list = []):
        self.top = top
        self.list = list
    
    def push(self, data):
        new_node = node(data)
        self.top = self.top + 1
        self.list[self.top] = new_node
    
    def pop(self):
        # Case of empty stack
        if self.top == -1:
            print("Stack is empty")
            return
        else:
            self.top = self.top - 1
            return self.list[self.top+1]
    
    def print_stack(self):
        for i in range(self.top+1):
            print(self.list[i].data)