class node():
    def __init__(self, data):
        self.data = data

#FIFO
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