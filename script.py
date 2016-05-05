# Jessica Jones
# CS 429
# Project 1

import os
import dec_tree

print ('Enter absolute path to file with training data,')
train_path = input('such as: /home/ML/proj1/data/training.txt\n').strip()
if not (os.path.isfile(train_path)):
  print ('Training data file path invalid; please start over')

test_path = input('\nEnter absolute path to file with testing data\n').strip()
if not (os.path.isfile(test_path)):
  print ('Testing data file path invalid; please start over')

val_path = input('\nEnter absolute path to file with validation data\n').strip()
if not (os.path.isfile(val_path)):
  print ('Validation data file path invalid; please start over')

output_csv = None
print ('\nTo output predictions, enter path to folder where output.txt should be created')
print ('such as /home/ML/proj1/')
csv_path = input('Else hit Enter\n').strip()
if csv_path:
  if not (os.path.isdir(csv_path)):
    print ('Output file path invalid; please start over')

dec_tree.do_id3(train_path, test_path, val_path, csv_path)