# -*- coding: utf-8 -*-

class MyOwnException(Exception):
    def __init__(self,message):
        super(MyTypeException,self).__init__()
        self.message = message

    def __str__(self,):
        return repr(self.message)



if __name__ == '__main__':
    a = [1,2,3]
    b = [2,3]
    try:
        if len(a) != len(b):
            print "a != b"
    except Exception,ex:
        print "Exception:",ex


    print "it is okay"