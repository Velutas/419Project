import sys
import os


movie_reviews_data_folder = "D:/CMPT/Anaconda/txt_sentoken"
#dataset = load_files(movie_reviews_data_folder, shuffle=False)

listoffiles = []

fneg = open('AllNeg.txt', 'w')
fneg.close()
fpos = open('AllPos.txt', 'w')
fpos.close()

fneg = open('AllNeg.txt', 'a')
fpos = open('AllPos.txt', 'a')

for root, dirs, files in os.walk(movie_reviews_data_folder):
    for file in files:
        if file.endswith('.txt'):
            listoffiles.append(file)
            #print(Location)
            


lenlist = len(listoffiles)

for i in range(0,lenlist):
    if i < 1000:
        Location = movie_reviews_data_folder + '/neg/' + listoffiles[i]
        file = open(Location, 'r')
        text = file.read()
        file.close()
        fneg.write(text)
    else:
        if i == 1000:
            print ("HERE")
        if i == 1999:
            print("DONE")
        Location = movie_reviews_data_folder + '/pos/' + listoffiles[i]
        file = open(Location, 'r')
        text = file.read()
        file.close()
        fpos.write(text)
        
f.close()
