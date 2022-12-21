import os
from textblob import TextBlob
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.collections as collections
import warnings
# Supress a warning due to a bug START
warnings.simplefilter(action='ignore', category=FutureWarning)
directory = os.getcwd()
# Supress a warning due to a bug END
# comes from: https://www.geeksforgeeks.org/how-to-get-file-extension-in-python/
# comes from https://www.geeksforgeeks.org/how-to-iterate-over-files-in-directory-using-python/
mainDataSet, speaker1DataSet, speaker2DataSet = [], [], []

for filename in os.listdir(directory):
    f = os.path.join(directory, filename)
    # checking if it is a file
    if os.path.isfile(f):
        # this will return a tuple of root and extension
        split_tup = os.path.splitext(filename)
        # extract the file name and extension
        file_name = split_tup[0]
        file_extension = split_tup[1]
        if file_extension == '.txt':
            # read text file into pandas DataFrame
            # https://stackoverflow.com/questions/47062493/how-can-i-get-python-to-read-every-nth-line-of-a-txt-file
            with open(file_name + '.txt') as f:
                lines = f.readlines()
                timestamp = lines[0::4]
                speaker = lines[1::4]
                text = lines[2::4]
                polarity, subjectivity = [], []
                #for a in speaker:
                #    if a == "Speaker 1\n":
                #        print(a)
                for phrase in text:
                    phraseSentiment = TextBlob(phrase)
                    polarity.append(phraseSentiment.polarity)
                    subjectivity.append(phraseSentiment.subjectivity)

                columns = {
                    "timeStamp": timestamp,
                    "speaker": speaker,
                    "polarity": polarity,
                    "subjectivity": subjectivity
                }
                data = pd.DataFrame(columns)
                data1 = data['index'] = range(1, len(data) + 1)
                data2 = data['base100'] = (data['index'] * 100) / len(data)
                speaker1DF = data[(data.speaker == "Speaker 1\n")]
                speaker1DataSet.append(speaker1DF)
                speaker2DF = data[(data.speaker == "Speaker 2\n")]
                speaker2DataSet.append(speaker2DF)


# Speaker 1 Polarity
fig, ax = plt.subplots()
fig.set_figwidth(27)
fig.set_figheight(10)
# ax.plot(x,y)
a = 0
for participant in speaker1DataSet:
    markerline, stemlines, baseline = plt.stem(speaker1DataSet[a]['base100'], speaker1DataSet[a]['polarity'], label="{} {}".format('Participant', a), linefmt="grey", markerfmt='D')
    markerline.set_markerfacecolor('blue')
    a += 1
#plt.legend()
ax.set_yscale("linear")
fig.subplots_adjust(left=0.04, right=0.98, bottom=0.1, top=0.9)
ax.grid(True, linestyle='-.')
plt.title("Interviewer Polarity")
#ax.tick_params(labelcolor='r', labelsize='medium', width=3)
plt.show()

# Speaker 1 Subjectivity
fig, ax = plt.subplots()
fig.set_figwidth(27)
fig.set_figheight(10)
# ax.plot(x,y)
a = 0
for participant in speaker1DataSet:
    markerline, stemlines, baseline = plt.stem(speaker1DataSet[a]['base100'], speaker1DataSet[a]['subjectivity'], label="{} {}".format('Participant', a), linefmt="--.", markerfmt='D')
    markerline.set_markerfacecolor('blue')
    a += 1
#plt.legend()
ax.set_yscale("linear")
fig.subplots_adjust(left=0.04, right=0.98, bottom=0.1, top=0.9)
ax.grid(True, linestyle='-.')
plt.title("Interviewer Sentiment")
#ax.tick_params(labelcolor='r', labelsize='medium', width=3)
plt.show()


# Speaker 2 Polarity
fig, ax = plt.subplots()
fig.set_figwidth(27)
fig.set_figheight(10)
# ax.plot(x,y)
b = 0
for participant in speaker1DataSet:
    plt.stem(speaker2DataSet[b]['base100'], speaker2DataSet[b]['polarity'], label="{} {}".format('Participant', b), linefmt="--.", markerfmt='D')
    b += 1
plt.legend()
ax.set_yscale("linear")
fig.subplots_adjust(left=0.04, right=0.98, bottom=0.1, top=0.9)
ax.grid(True, linestyle='-.')
#ax.tick_params(labelcolor='r', labelsize='medium', width=3)
plt.title("Designer Persona Polarity")
plt.show()

# Speaker 2 Subjectivity
fig, ax = plt.subplots()
fig.set_figwidth(27)
fig.set_figheight(10)
# ax.plot(x,y)
b = 0
for participant in speaker1DataSet:
    plt.stem(speaker2DataSet[b]['base100'], speaker2DataSet[b]['subjectivity'], label="{} {}".format('Participant', b), linefmt="--.", markerfmt='D')
    b += 1
plt.legend()
ax.set_yscale("linear")
fig.subplots_adjust(left=0.04, right=0.98, bottom=0.1, top=0.9)
ax.grid(True, linestyle='-.')
#ax.tick_params(labelcolor='r', labelsize='medium', width=3)
plt.title("Designer Persona Sentiment")
plt.show()