import serial #Serial imported for Serial communication
import time #Required to use delay functions
import math

ser = serial.Serial('/dev/cu.usbmodem14101',9600,timeout=1)

time.sleep(2)
#print(ser.readline())

#light = 0
try:
    i = 0
    while True:
        #time.sleep(1)
        #print('1')
        #ser.write(b'1')
        #time.sleep(1)
        #print('0')
        #ser.write(b'0')
        #try:
        #    ser.open()
        #except:
        #    None
        #ser.write(b'hi\n')
        #print("read", ser.readline())
        i+=1
        '''
        val = str(int(math.sin(i/10)*5+5))
        print(val)
        ser.write(val.encode())
        print("write", val.encode())
        '''
        num = int(math.sin(i/50)*500+500)
        #if num<0:
        #    num=0
        val = bytearray([int(num/100), int((num%100)/10), (num%100)%10])
        print("wrote", val)
        ser.write(val)
        #ser.write(b'\n')
        #ser.close()
        #ser.write('2'.encode())
        #print("write", '2'.encode())
        time.sleep(0.02)
        text = ser.readline()
        print("read", text)
        #for i in range(4):
            #ser.write(b'0')
            #ser.write(b'%d' % i)
            #ser.write(b'\n')
            #text = ser.readline()
            #print(text)
            ##ser.write(str(i).encode())
            ##time.sleep(0.5)
            #ser.write(b'1')
            #time.sleep(0.5)
            #text = ser.read()
            #print(text)
except KeyboardInterrupt:
    None
ser.close()
print("Program execution complete")
