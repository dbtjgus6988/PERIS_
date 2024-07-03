import csv
import datetime
import sys
import numpy as np

file = sys.argv[1]
output = "ratings_Userbehavior.csv"

with open(file,"r", newline = '') as f, open(output, 'w', newline ="") as f_out:
    reader = csv.reader(f, delimiter =',')
    writer = csv.writer(f_out)

    for row in reader:
        if (1511577600 < int(row[3])) and (int(row[3])<1512259200):
            user_id = row[0]
            item_id = row[1]
            rating = float(row[2])
            timestamp = row[3]
            new = [user_id, item_id, rating, timestamp]
            writer.writerow(new)

    
    
