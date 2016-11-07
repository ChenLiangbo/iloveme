#!python
#coding:utf-8
import math

def isPrime(num):
    flag  = True
    int_n = math.ceil(math.sqrt(num))   
    for i in range(2,int(int_n)):
        imod = num % i       
    	if imod ==0:
    		flag = False
    		break
    return flag

def getPrime(num):
    '''
    find all prime in range of num
    '''    
    primlist = []    
    for i in range(2,int(num)):
        flag = isPrime(i)    
        if flag == True:
            primlist.append(i)
    return primlist


def func(p,q):
    '''
    return (q-1)(p-1)
    '''    
    return (p-1)*(q-1)


def get_d(e,q,p):
    '''
    (d x e) % (q-1)(p-1) = 1;e < d < fn
    '''
    d = 0
    fn = (p-1)*(q-1)
    for i in range(1,fn):
        dmod = (i*e) % fn
        if dmod == 1:
            d = i
            break    
    return d

def RSANumberEncode(c,gkey):
    # m = c**e % n
    e = gkey[0]
    n = gkey[1]
    tmp = c**e
    m = tmp % n
    return m

def RSANumberDecode(m,skey):
    d = skey[0]
    n = skey[1]
    tmp = m**d
    c = tmp % n
    return c

def rsaNumber(num,key):
    n = key[1]
    de = key[0]
    tmp = num**de
    codenum = tmp % n
    return codenum

def RSAEncode(mString,gkey):
    '''
    rsa encode method 
    input string conment,e,n
    output coding content key = (e,n)
    '''   
    e = gkey[0]
    n = gkey[1]
    codeList = []
    mystrlen = len(mString)
    for i in range(mystrlen):
        m = ord(mString[i])        #get char's interger       
        c = RSANumberEncode(m,gkey)        
        codeList.append(c)   
    return codeList

def RSADecode(rsaEcodeList,skey):
    '''
    rsa decode method
    input RSAencode string conment cString,d,n
    output decode string conment mString  skey = (d,n)
    '''
    d = skey[0]
    n = skey[1]
    decode_string = ""
    enlen = len(rsaEcodeList)
    decodeList =[]
    for m in rsaEcodeList:       
        decode = RSANumberDecode(m,skey)    
        decodeList.append(chr(int(m)))
        decode_string = decode_string + chr(decode)
    return decode_string

# def RSAString(string,key):
#     '''
#     rsa coding and decoding method for string
#     input string and key 
#     output string whatever encoding or decoding 
#     '''
#     print "--------running in RSAString--------"
#     de = key[0]
#     n  = key[1]
#     stringLen   = len(string)
#     code_string = ""
#     for i in range(stringLen):
#         strchar  = string[i]
#         print "strchar is ",strchar
#         asciinum = ord(strchar)     #ascii num
#         rsanum   = (asciinum**de) % n #rsa coding for ascii num 
#         tmpchar  = chr(rsanum)        #rsa coding char from coding num
#         code_string = code_string + tmpchar
#     return code_string

def RSAString(string,key):
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



prime = getPrime(100)
# print prime

#init select p,q
# p = 431
p=19
# q = 941
q=7
n = p*q
fn = func(p,q)
e = 23  #e 与 fn 互质
d = get_d(e,q,p)
Ku = (e,n)    ##公钥
Kr = (d,n)    ##私钥
string = "hk1688"
ecode1   = RSAEncode(string,Ku) 
decode1  = RSADecode(ecode1,Kr)
print "get_d = %d" % d
print "--------RSAEncode---RSADecode -------"
print "string after rsa method -------",ecode1
print "decoding string is ----",decode1
# ecode2  =RSAString(string,Ku) 
# decode2 = RSAString(ecode1,Kr)
# print "-------RSAString--------"
# print "string after rsa method -------",code2
# print "decoding string is ----",decode2
mnum = RSANumberEncode(121,Ku)
cnum = RSANumberDecode(mnum,Kr)
print "num encode is :",mnum
print "num decode is :",cnum
print "it is ok!"  
# for i in range(len(string)):
#     char = string[i]
#     print "char in string is:",chr(char)
