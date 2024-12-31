print("I am hungry!")
class Man:
    def __init__(self,name):
        self.name = name
        self.age = 0
        print("Initialized!")
    def hello(self):
        print("hello "+self.name)

m = Man("kk")
m.hello()
m.age = 10
print(m.age)