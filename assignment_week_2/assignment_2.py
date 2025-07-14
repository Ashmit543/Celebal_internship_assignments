class Node:
    def __init__(self, data):
        self.data = data
        self.next = None


class LinkedList:
    def __init__(self):
        self.head = None

    def add_to_end(self, data):
        new_node = Node(data)
        if not self.head:
            self.head = new_node
            return

        current = self.head
        while current.next:
            current = current.next
        current.next = new_node

    def print_list(self):
        if not self.head:
            print("List is empty.")
            return

        current = self.head
        while current:
            print(current.data, end=" -> ")
            current = current.next
        print("None")

    def delete_nth_node(self, n):
        if n <= 0:
            raise ValueError("Index must be a positive integer.")

        if not self.head:
            raise IndexError("Cannot delete from an empty list.")

        if n == 1:
            self.head = self.head.next
            return

        current = self.head
        count = 1
        while current and count < n - 1:
            current = current.next
            count += 1

        if not current or not current.next:
            raise IndexError("Index out of range.")

        current.next = current.next.next


# Sample usage
if __name__ == "__main__":
    ll = LinkedList()
    ll.add_to_end(10)
    ll.add_to_end(20)
    ll.add_to_end(30)
    ll.add_to_end(40)

    print("Initial List:")
    ll.print_list()

    try:
        ll.delete_nth_node(2)
        print("\nList after deleting 2nd node:")
        ll.print_list()

        ll.delete_nth_node(10)  # Index out of range
    except Exception as e:
        print(e)

    try:
        empty_list = LinkedList()
        empty_list.delete_nth_node(1)  # Deleting from empty list
    except Exception as e:
        print(e)
