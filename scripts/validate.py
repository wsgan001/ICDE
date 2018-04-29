print("===========NAME===========")
filepath = 'tf_files/name_info.txt'  
with open(filepath) as fp:  
   line = fp.readline()
   while line:
       print("{}".format(line.strip("\n")))
       line = fp.readline()

print("===========DATE===========")
filepath = 'tf_files/date_info.txt'  
with open(filepath) as fp:  
   line = fp.readline()
   while line:
       print("{}".format(line.strip("\n")))
       line = fp.readline()