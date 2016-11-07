#!user/bin/env/python
#coding:utf-8
import math
import hashlib

class ChenlbEncoder(object):
    '''RSA encoder and decoder for a number,list and string
       RSA and hashlib.sha1 method encoder
       Given p,q when initialize,add e and can be used
    ''' 

    def __init__(self,p,q,e):
        super(ChenlbEncoder,self).__init__()      
        if (self.isPrime(p) and self.isPrime(q)) is False:
            raise ValueErro('Both p and q should be prime!!')
        self.q = q
        self.p = p
        self.e = e
        self.n = p*q
        self.fn = self.get_fn()
        self.d = self.get_d()

    def isPrime(self,num):    
        if num <= 1:    
            return False   
        if num == 2:    
            return True   
        if num % 2 == 0:    
            return False   
        i = 3   
        while i * i <= num:    
            if num % i == 0:    
                return False   
            i += 2   
        return True   


    def getPrime(self,num):
        '''find all prime in range of given num
        '''    
        primlist = []    
        for i in range(2,int(num)):
            flag = self.isPrime(i)    
            if flag == True:
                primlist.append(i)
        return primlist

    def get_fn(self):
        '''return (q-1)(p-1) '''    
        return (self.p-1)*(self.q-1)


    def get_d(self):
        '''(d x e) % (q-1)(p-1) = 1;e < d < fn'''        
        d = 0        
        for i in range(1,self.fn):
            dmod = (i*self.e) % self.fn
            if dmod == 1:
                d = i
                break    
        return d

    def RSANumberEncode(self,c):
        '''c means number need to be encoded'''
        # m = c**e % n       
        tmp = c**self.e
        m = tmp % self.n
        return m

    def RSANumberDecode(self,m):
        '''m means encoded by RSANumberEncoder'''      
        tmp = m**self.d
        c = tmp % self.n
        return c

    def rsaNumber(self,num,key):
        '''Both can encode and decode by type of key tuple'''
        n = key[1]
        de = key[0]
        tmp = num**de
        codenum = tmp % n
        return codenum

    def RSAStringEncode(self,mString):
        '''
        rsa encode method 
        input string conment,e,n
        output coding content key = (e,n)
        '''           
        codeList = []
        encoderString = ''
        mystrlen = len(mString)
        for i in range(mystrlen):
            m = ord(mString[i])        #get char's interger       
            c = self.RSANumberEncode(m)        
            codeList.append(c)   
        return codeList

    def RSAStringDecode(self,rsaEcodeList):
        '''
        rsa decode method
        input RSAencode string conment cString,d,n
        output decode string conment mString  skey = (d,n)
        '''    
        enlen = len(rsaEcodeList)
        decodeList =[]
        decode_string = ''
        for m in rsaEcodeList:       
            decode = self.RSANumberDecode(m)    
            decodeList.append(chr(int(m)))
            decode_string = decode_string + chr(decode)
        return decode_string


    def RSAString(self,string,key):
        '''
        rsa coding and decoding method for string
        input string and key 
        output string whatever encoding or decoding 
        '''    
        de = key[0]
        n  = key[1]
        stringLen   = len(string)
        code_string = ""
        for i in range(stringLen):
            strchar  = string[i]
            asciinum = ord(strchar)     #ascii num
            rsanum   = (asciinum**de) % n #rsa coding for ascii num 
            tmpchar  = chr(rsanum)        #rsa coding char from coding num
            code_string = code_string + tmpchar
        return code_string

    def chenEncoder(self,string):
        '''use sha1 encoder a string,return its hex digest'''
        m = hashlib.sha1()
        m.update(string)
        return m.hexdigest()

    def chenCheckHexdigest(self,cString,mString):
        '''check string encoded by hashlib.sha1,return flag
            cString is comment
            mString is hex digest
        '''
        m = hashlib.sha1()
        m.update(cString)
        if m.hexdigest() == mString:
            return True
        else:
            return False

if __name__ == '__main__':
    
    p=19
    q=7
    n = p*q
    e = 23  #e 与 fn 互质
    chenCoder = ChenlbEncoder(p,q,e)
    prime = chenCoder.getPrime(100)
    print "prime = ",prime
    
    
    Ku = (chenCoder.e,chenCoder.n)    ##公钥
    Kr = (chenCoder.d,chenCoder.n)    ##私钥
    string = "hk1688"   
  
    ecode2  =chenCoder.RSAStringEncode(string) 
    decode2 = chenCoder.RSAStringDecode(ecode2)
    print "-------RSAString--------"
    print "encode2 -------",ecode2
    print "decode2 -------",decode2  
    test3 = "chenliangbohk1688"
    code3 = chenCoder.RSAString(test3,Ku)
    dcode3 = chenCoder.RSAString(code3,Kr)
    print "-----------------RSAString-------------------"
    print "code3 = ",code3
    print "dcode3 = ",dcode3

    hexdigest = chenCoder.chenEncoder('chenliangbohk1688')
    print chenCoder.chenCheckHexdigest('chenliangbohk1688',hexdigest)
    print "it is ok!"  
 
