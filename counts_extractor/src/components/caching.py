from typing import Optional, Union
from src.components.models import Track

class Node:
    def __init__(self, key: int, val: Track, left: 'Optional[Node]' = None, right: 'Optional[Node]' = None) -> None:
        """
        Initializes a Node with the provided key, value, left and right nodes.

        Parameters:
            key (int): The key of the Node.
            val (Track): The value associated with the Node.
            left (Optional[Node], optional): The left child Node. Defaults to None.
            right (Optional[Node], optional): The right child Node. Defaults to None.

        Returns:
            None
        """
        self.val = val
        self.key = key
        self.left = left
        self.right = right

class LRUCache:
    def __init__(self, capacity: int):
        """
        Initializes the LRUCache with the specified capacity.

        Parameters:
            capacity (int): The maximum number of elements that the LRUCache can hold.

        Returns:
            None
        """
        self.head = Node(-1,-1)
        self.tail = Node(-1,-1)
        self.cap = capacity
        self.map = dict()

        self.head.right = self.tail
        self.tail.left = self.head

    def get(self, key: int) -> Union[Track, None]:
        try:
            node = self.map[key]
            res = node.val
            self.remove_node(node)
            self.add_node(node)

            return res
        except KeyError:
            return None

    def put(self, key: int, value: Track) -> None:

        try:
            node = self.map[key]
            node.val = value
            self.remove_node(node)

        except KeyError:
            if len(self.map) == self.cap:
                self.remove_node(self.tail.left)
        
        self.add_node(Node(val=value, key=key))

        

    def add_node(self, node: Node) -> None:
        self.map[node.key] = node

        node.right = self.head.right
        node.right.left = node
        node.left = self.head
        self.head.right = node

    def remove_node(self, node: Node) -> None:
        del self.map[node.key]
        
        del_left = node.left
        del_right = node.right
        del_left.right = node.right
        del_right.left = node.left

        node.right = None
        node.left = None

