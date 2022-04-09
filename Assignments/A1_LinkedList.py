class node():
    def __init__(self, data,next=None):
        self.data = data
        self.next = next

class LinkedList():
    def __init__(self, head=None):
        self.head = head

    def add_node(self, data):
        new_node = node(data)
        # Case of empty LL
        if self.head is None:
            self.head = new_node
        else:
            p = self.head
            while p.next is not None:
                p = p.next
            p.next = new_node
        
    def delete_node(self, data):
        p = self.head
        # If head is deleted
        if p.data == data:
            self.head = p.next
            return
        # Otherwise
        while p.next is not None:
            if p.next.data == data:
                p.next = p.next.next
                return
            p = p.next
        # If it is not returned, that means node not found, so we will print error
        print("Node not found")

    def print_list(self):
        p = self.head
        while p is not None:
            print(p.data)
            p = p.next