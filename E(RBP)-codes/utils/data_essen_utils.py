import music21
from music21 import *
import os


#path = path to data file
paths = []
pitches = []
final_pitches = []
for fname in os.listdir(path):
	if fname.endswith('.mid'):
		paths.append(os.path.join(path,fname))

for path in paths:
    pitches.append(converter.parse(path))

for path in pitches:
	for pitch in path:
        final_pitches.extend(pitch.pitches)

#store all pitches in a file

file = open('~/data/data.txt', 'w')

for pitch in final_pitches:
	print >> file, pitch
