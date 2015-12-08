import sys
import os


movie_reviews_data_folder = sys.argv[1]
#dataset = load_files(movie_reviews_data_folder, shuffle=False)

listoffiles = []

f = open('AllText.txt', 'w')
f.close
f = open('AllText.txt', 'a')

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
        f.write(text)
    else:
        if i == 1000:
            print ("HERE")
        if i == 1999:
            print("DONE")
        Location = movie_reviews_data_folder + '/pos/' + listoffiles[i]
        file = open(Location, 'r')
        text = file.read()
        file.close()
        f.write(text)
        
f.close()
