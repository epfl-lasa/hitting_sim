class Object:
    def __init__(self, mass):
        self.mass = mass


class Box(Object):
    def __init__(self, size, mass):
        self.l = size[0]  # length
        self.b = size[1]  # breadth
        self.h = size[2]  # height
        Object.__init__(self, mass)


class Sphere(Object):
    def __init__(self, r, mass):
        self.r = r  # radius
        Object.__init__(self, mass)