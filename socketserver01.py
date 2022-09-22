# tcpclient.py

from socket import *
import pickle
import time

ip='203.252.164.24'
port=8608

clientSocket=socket(AF_INET, SOCK_STREAM)
clientSocket.connect((ip,port))

def listsend(cmd, client):
    sentence="".join(map(str,cmd))
    print(sentence)
    clientSocket.sendall(sentence.encode("utf-8"))
    return 0
      
print("Connect Success")

data=[]
l=1

while True :	
    data = [0,1,0,1,0,1,1,0,1,1]
    listsend(data, clientSocket)
    time.sleep(0.3)
    
clientSocket.close()
