class tester:
    def __init__(self):
        pass
    
    def fcn_one(self, a):
        return fcn_one(a)
    
def fcn_one(a):
    return a + 20

if __name__ == "__main__":
    obj = tester()
    b = 30
    print(fcn_one(b))
    print(obj.fcn_one(b))