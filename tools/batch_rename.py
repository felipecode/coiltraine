import os
import re
count =0
file_list = sorted(os.listdir("SeqTotal"))

print file_list

for filename in file_list:


        print filename
        newfilename = "SeqTotal/data_"+str(count).zfill(5) + ".h5"

        os.rename("SeqTotal/" + filename,newfilename )
        count +=1