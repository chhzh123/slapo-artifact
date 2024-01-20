import sys
from os import listdir

files = listdir(sys.argv[1])

for f in files:
    if f.endswith('.schedule'):
        with open(sys.argv[1] + f) as fp:
            lines = fp.readlines()
            new_lines = []
            for line in lines:
                if "for" in line:
                    continue
                new_lines.append(line)
            print(f + ": " + str(len(new_lines)))
