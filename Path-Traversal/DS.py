class node():
    def __init__(self, data):
        self.data = data

class Queue():
    def __init__(self, front = 0, rear = 0, list = []):
        self.front = front
        self.rear = rear
        self.list = list
    
    def enqueue(self, data):
        new_node = node(data)
        self.rear = self.rear + 1
        self.list[self.rear] = new_node
    
    def dequeue(self):
        # Case of empty queue
        if self.front == self.rear:
            print("Queue is empty")
            return
        else:
            self.front = self.front + 1
            return self.list[self.front-1]

    def print_queue(self):
        for i in range(self.front, self.rear+1):
            print(self.list[i].data)

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