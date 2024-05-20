import numpy as np

def pathtracker_shapes(nb_objects, idx=0):
    list_shapes = []
    for obj in range(nb_objects):
        #randomly choose between squares and bigsquare and bigsquare2
        if idx == 0:
            list_shapes.append(Squares())
        elif idx == 1:
            list_shapes.append(BigSquares())
        else:
            list_shapes.append(BigSquares2())
    return list_shapes

def start_shapes(nb_objects):
    list_shapes = []
    for obj in range(nb_objects):
        list_shapes.append(Shape())
    return list_shapes

def end_shapes(nb_objects, shapes):
    for obj in range(nb_objects):
        shapes[obj].end()
    return shapes

def blobs():
    blob = np.zeros((5,5))
    #fill the perimeter of the blob
    blob[0, :] = 1
    blob[-1, :] = 1
    blob[:, 0] = 1
    blob[:, -1] = 1
    return blob

def frames_shapes(nb_objects, shapes):
    for obj in range(nb_objects):
        shapes[obj].update()
    return shapes

class Squares:
    def __init__(self):
        self.n = 5
        self.grid = np.zeros((self.n, self.n))
        self.grid[1:4, 1:4] = 1

class BigSquares:
    def __init__(self):
        self.n = 5
        self.grid = np.zeros((self.n, self.n))
        self.grid[2, 2] = 1

class BigSquares2:
    def __init__(self):
        self.n = 5
        self.grid = np.zeros((self.n, self.n))
        self.grid[2, 2] = 1
        self.grid[0,1:4] = 1
        self.grid[-1,1:4] = 1
        self.grid[1:4,0] = 1
        self.grid[1:4,-1] = 1

class Shape:
    def __init__(self):
        self.n = 5
        self.grid = np.zeros((self.n, self.n))
        self.grid[1:4, 1:4] = np.random.randint(0, 2, (3, 3))

    def update(self):
        new_grid = np.zeros_like(self.grid)
        for i in range(self.n):
            for j in range(self.n):
                #count neighbors of each pixel
                neighbors_sum = np.sum(self.grid[max(0, i - 1):min(self.n, i + 2), max(0, j - 1):min(self.n, j + 2)]) - self.grid[i, j]

                # if self.grid[i, j] == 1 and (neighbors_sum < 2 or neighbors_sum > 3): #neighbors_sum < 2 or
                #     new_grid[i, j] = 0
                if self.grid[i, j] == 0 and neighbors_sum == 3:
                    new_grid[i, j] = 1
                else:
                    new_grid[i, j] = self.grid[i, j]
        self.grid = new_grid
        if np.sum(self.grid) == 0:
            #self.grid[2, 2] = 1
            self.grid[1:4, 1:4] = np.random.randint(0, 2, (3, 3))

    def end(self):
        self.update()
        if np.sum(self.grid) == 0:
            self.grid[1:4, 1:4] = np.random.randint(0, 2, (3, 3))
        self.grid[0, :] = 0
        self.grid[-1, :] = 0
        self.grid[:, 0] = 0
        self.grid[:, -1] = 0