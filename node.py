# Jessica Jones
# CS 429
# Project 1

from itertools import count

class Node:

  # count of nodes generated for use in debugging 
  count = 0

  def __init__(self, parent):
    # parent is parent node
    self.parent = parent
    self.children = []
    self.label = None
    self.split_attrib = None
    self.attributes = {}
    self.examples = []
    self.count = self.__class__.count
    self.__class__.count += 1 

