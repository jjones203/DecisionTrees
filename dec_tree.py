# Jessica Jones
# CS 429
# Project 1

import csv
import math
import numpy
import node
import mushroom

# info about mushroom dataset
target_attrib = mushroom.target_attrib
positive = mushroom.positive
negative = mushroom.negative
null_class = mushroom.unknown_class
unknown_val = mushroom.unknown_val
fields = mushroom.attributes
attribute_values = mushroom.attribute_values

# info for table of chi-squared critical values
chi_sq_path = './chi_squared_table.txt'
chi_sq_vals = ['dof', '50', '95', '99']
chi_squared_table = []

# print tree for debugging
def print_tree(root, level):
  if root.parent:
    parent_name = str(root.parent.split_attrib)
  else:
    parent_name = ''
  if root.label: 
    print("#",root.count," Level-",level," Label-",root.label," Parent-",parent_name," ",str(root.attributes))
  else:
    print("#",root.count," Level-",level," Split attrib-",root.split_attrib," Parent-",parent_name," ",str(root.attributes))
  for child in root.children:
    print_tree(child, level+1)

# get depth of tree
def get_max_depth(root, level):
  if root.label:  
    return level
  else:
    max_depth = 0
    for child in root.children:  
      depth = get_max_depth(child, level+1)
      if depth > max_depth:
        max_depth = depth
    return max_depth

# get number nodes in tree
def get_num_nodes(root):
  if root == None:
    return 0
  else:
    count = 1
    for child in root.children:
      count += get_num_nodes(child)
    return count

# when classifying instance with missing value for an attribute,
# returns token value at leaf; method from Quinlan 1986
def get_leaf_token(node, instance, token):
  # return token value after reaching leaf
  if node.label:
    if node.label == positive:
      return token
    else:
      return -token
  # if not at leaf, find attribute node splits on. Multiply token
  # by percent of training examples that share instance's attribute value.
  # Then recurse.
  else:
    split_attrib = node.split_attrib
    instance_value = instance[split_attrib]
    percent_match = len(search_examples(split_attrib, instance_value, node.examples))/len(node.examples)
    token *= percent_match
    child_match = [child for child in node.children if child.attributes[split_attrib] == instance_value][0]
    return get_leaf_token(child_match, instance, token)

# takes list of instances, classifies each and
# returns list of classifications
def classify_instance_list(root, target_attrib, instance_list):
  class_list = []
  for instance in instance_list:
    classification = classify_instance(root, instance)
    class_list.append(classification)
  return class_list

# classifies an instance using decision tree
def classify_instance(root, instance):
  # if root has label, return label
  if root.label:
    return root.label
  # else get root's split attribute
  else:
    split_attrib = root.split_attrib
    # if value of attribute unknown, call other function to get probable class
    # assume an instance does not have unknown values for > 1 attribute
    if instance[split_attrib] == unknown_val:
      return classify_unknown_attrib(root,instance)
    # check which of root's children's value of split attribute matches instance's value
    for child in root.children:
      if child.attributes[split_attrib] == instance[split_attrib]:
        # recurse
        return classify_instance(child, instance)

# for instances with an unknown attribute value,
# return probable class of instance by passing token value through tree
# method from Quinlan 1986
def classify_unknown_attrib(root, instance):
  # pos_token and neg_token sum values returned from leaves
  pos_token = 0
  neg_token = 0
  num_root = len(root.examples)
  # pass token to each child proportional to the number of
  # training examples at root that went to each child
  for child in root.children:
    num_children = len(child.examples)
    leaf_val = get_leaf_token(child, instance, num_children/num_root)
    # if token value is positive, returned from leaf labeled +
    if leaf_val > 0:
      pos_token += leaf_val
    else:
      neg_token -= leaf_val
  if pos_token > neg_token:
    return positive
  else:
    return negative
     
# return list of examples for whom attrib == value
def search_examples(attrib, value, examples):
  matches = [ex for ex in examples if ex[attrib] == value]
  #print (len(matches))
  return matches

# get the most common class from set of examples
def get_most_common_value(target_attrib, examples):
  num_examples = len(examples)
  num_positive = len(search_examples(target_attrib, positive, examples))
  if num_positive > (num_examples - num_positive):
    return positive
  else:
    return negative

# calculates classification error
def get_classification_error(prob_positive, prob_negative):
  classification_error = 1 - max(prob_positive, prob_negative)
  return classification_error

# calculates entropy
def get_entropy(prob_positive, prob_negative):
  entropy = 0
  if prob_positive:
    entropy += - prob_positive * math.log(prob_positive,2)
  if prob_negative:
    entropy += - prob_negative * math.log(prob_negative,2)
  return entropy

# calculate resulting impurity for splitting on given attribute
def eval_attrib(examples, target_attrib, attrib_to_eval, criterion):
  num_examples = len(examples)
  impurity = 0
  unknowns = None
  # if value of target_attrib is '?' for any example
  if (search_examples(attrib_to_eval, unknown_val, examples)):
  # get number of '?' in positive and negative classes
    unknowns = search_examples(attrib_to_eval, unknown_val, examples)
    num_unknowns = len(unknowns)
    num_pos_unknowns = len(search_examples(target_attrib, positive, unknowns))
    num_neg_unknowns = len(search_examples(target_attrib, negative, unknowns))
  # for each valid value of target_attrib
  # get number of positive & negative instance, adjusting for
  # unknowns if necessary
  for val in attribute_values[attrib_to_eval]:
    examples_val = search_examples(attrib_to_eval, val, examples)
    if unknowns:
      raw_num_pos = len(search_examples(target_attrib, positive, examples_val))
      raw_num_neg = len(search_examples(target_attrib, negative, examples_val))
      adj_ratio = (raw_num_pos+raw_num_neg)/(num_examples-num_unknowns)
      # from Quinlan: pi = pi + pu * (pi + ni)/(sum(pi + ni))
      num_positive = raw_num_pos + num_pos_unknowns * adj_ratio
      num_negative = raw_num_neg + num_neg_unknowns * adj_ratio
    else:
      num_positive = len(search_examples(target_attrib, positive, examples_val))
      num_negative = len(search_examples(target_attrib, negative, examples_val))
    # calculate prob(+) and prob(-); need to avoid / by 0 error
    if (num_positive+num_negative):
      prob_positive = num_positive/(num_positive+num_negative)
    else:
      prob_positive = 0
    prob_negative = 1 - prob_positive
    # calculate entropy or classification error per criterion
    if criterion == 'entropy':
      impurity_val = get_entropy(prob_positive, prob_negative)
    else:
      impurity_val = get_classification_error(prob_positive, prob_negative)
    # calculate weighted sum of entropy/classification
    impurity += impurity_val * (num_positive+num_negative)/num_examples
  # return weighted sum
  return impurity

# decide which of available attributes to split on
def get_best_attrib(examples, target_attrib, attributes, criterion):
  start_impurity = 0
  num_positive = len(search_examples(target_attrib, str(positive), examples))
  prob_positive = num_positive/len(examples)
  if (criterion == 'entropy'):
    start_impurity = get_entropy(prob_positive, 1-prob_positive)
  else:
    start_impurity = get_classification_error(prob_positive, 1-prob_positive)
  best_attrib = None
  best_gain = 0
  for attrib in attributes:
    impurity = eval_attrib(examples, target_attrib, attrib, criterion)
    gain = start_impurity - impurity
    if gain > best_gain:
      best_gain = gain
      best_attrib = attrib
  return best_attrib

# get chi-squared value for attribute chosen for split;
# formula and variable names from:
# http://isites.harvard.edu/fs/docs/icb.topic539621.files/lec7.pdf
def get_chi_squared(examples, target_attrib, attrib_for_split):
  # p & n are number of +/- examples prior to splitting on node
  p = len(search_examples(target_attrib, positive, examples))
  n = len(search_examples(target_attrib, negative, examples))
  devX = 0
  for x in attribute_values[attrib_for_split]:
    # Dx is subset of examples with split_attrib == x
    Dx = search_examples(attrib_for_split, x, examples)
    if len(Dx) == 0:
      continue
    # p_hat & n_hat are number of +/- examples in Dx if
    # Dx has same distribution as all examples in node
    p_hat = p/(p+n) * len(Dx)
    n_hat = n/(p+n) * len(Dx)
    # px and nx are actual number of +/- examples in Dx
    px = len(search_examples(target_attrib, positive, Dx))
    nx = len(search_examples(target_attrib, negative, Dx))
    # devX is deviation from absence of pattern
    devX += ((px - p_hat) ** 2)/p_hat + ((nx - n_hat) ** 2)/n_hat
  return devX

# calculate threshold value for chi-squared test
def get_threshold(attrib_for_split, confidence):
  # get degrees of freedome
  deg = len(attribute_values[attrib_for_split])-1
  # get critical values for attribute's degrees of freedom
  critical_vals = [row for row in chi_squared_table if row['dof'] == str(deg)][0]
  return float(critical_vals[str(confidence)])

# helper method labels node with most common class of its examples
def label_most_common(node, target_attrib):
  if (get_most_common_value(target_attrib, node.examples)) == positive:
    node.label = positive
    node.attributes[target_attrib] = positive
  else:
    node.label = negative
    node.attributes[target_attrib] = negative
  return node

# ID3 from Mitchell
def make_id3_tree(examples, target_attrib, attributes, parent, inherited_attributes, criterion, confidence):
  root = node.Node(parent)
  # Node.attributes represents conjunction of attribute tests which are true
  # for training examples at node
  root.attributes = inherited_attributes.copy()
  if parent:
    parent.children.append(root)
  root.examples = examples[:]
  # if all examples are +, return single-node tree Root with label +
  if len(search_examples(target_attrib, positive, examples)) == len(examples):
    root.label = positive
    root.attributes[target_attrib] = positive
  # if all examples are -, return single-node tree Root with label -
  elif len(search_examples(target_attrib, negative, examples)) == len(examples):
    root.label = negative
    root.attributes[target_attrib] = negative
  # if set of attributes to be tested is empty, return single-node 
  # tree Root with label == most common value of target_attrib in examples
  elif not attributes:
    root = label_most_common(root, target_attrib)
  else:
    # else
    # find attribute A w/highest info gain/accuracy
    attrib_for_split = get_best_attrib(examples, target_attrib, attributes, criterion)
    # perform chi-squared test on attribute A if confidence level entered
    if confidence:
      chi_sq = get_chi_squared(examples, target_attrib, attrib_for_split)
      threshold = get_threshold(attrib_for_split, confidence)
      if chi_sq < threshold:
        root = label_most_common(root, target_attrib)
    # if root not yet labeled, either passed chi-square test or wasn't tested
    if not root.label:
      # set decision attribute for root == A
      root.split_attrib = attrib_for_split
      # for each possible value vi of A
      for value in attribute_values[attrib_for_split]:
        # add a branch below root corresponding to A == vi
        # let examples_vi be subset of examples with A == vi
        examples_val = search_examples(attrib_for_split, value, examples)
        # if |examples_vi| == 0, add leaf node w/most common value of target_attrib in examples
        if not examples_val and not root.children:
          child = node.Node(parent)
          root.children.append(child)
          label = get_most_common_value(target_attrib,examples)
          child.label = label
          child.attributes = root.attributes.copy()
          child.attributes[target_attrib] = label
        # else add subtree make_id3_tree(examples_vi, target_attrib, (attributes-A), criterion, root)
        else:
          attributes_child = attributes[:]
          attributes_child.remove(attrib_for_split)
          attribute_values_child = root.attributes.copy()
          attribute_values_child[attrib_for_split] = value
          make_id3_tree(examples_val, target_attrib, attributes_child, root, attribute_values_child, criterion, confidence)
  return root  

# populate lists of dictionaries from csv files
def get_file_data(filename, filefields, data):
  with open(filename) as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
      data.append(dict(zip(filefields, row)))
  csvfile.close()

# output list of classes to csv
def output_csv(filepath, class_list):
  filename = filepath +'output.txt'
  with open(filename, 'w') as csvfile:
    writer = csv.writer(csvfile)
    for item in class_list:
      writer.writerow([item,])
  csvfile.close()

# generate confusion matrix for instances w/known classes
def get_confusion_mtx(target_attrib, instances, predictions):
  mtx = numpy.zeros((2,2))
  num_instances = len(instances)
  if num_instances != len(predictions):
    print ('Unequal number of instances and predictions')
  else:
    for i in range(0,len(instances)):
      # find if true pos/neg, false pos/neg  
      gt = instances[i][target_attrib]
      pred = predictions[i]
      # increment confusion mtx
      if gt == pred:
        if gt == positive:
          mtx[0][0] += 1
        else:
          mtx[1][1] += 1
      else:
        if gt == positive:
          mtx[0][1] += 1
        else:
          mtx[1][0] += 1  
  return mtx

def get_tree(training_data, confidence, entropy):
  # attributes = attributes available to split on
  attributes = fields[:]
  attributes.remove(target_attrib)
  if entropy:
    criterion = 'entropy'
  else:
    criterion = 'class_error'
  tree_root = make_id3_tree(training_data, target_attrib, attributes, None, {}, criterion, int(confidence))
  return tree_root

def test_tree(tree_root, results_table, column, test_data):
    predictions = classify_instance_list(tree_root, target_attrib, test_data)
    conf_mtx = get_confusion_mtx(target_attrib, test_data, predictions)
    print (conf_mtx.astype(int))
    accuracy = numpy.sum(conf_mtx.diagonal())/numpy.sum(conf_mtx)
    depth = get_max_depth(tree_root, 0)
    num_nodes = get_num_nodes(tree_root)
    results_table[1][column] = str(int(accuracy*100))+'%'
    results_table[2][column] = str(num_nodes)
    results_table[3][column] = str(depth)


def do_id3(train_file, test_file, validation_file, output_path):
  training_data = []
  get_file_data(train_file, fields, training_data)
  get_file_data(chi_sq_path, chi_sq_vals, chi_squared_table)
  test_data = []
  get_file_data(test_file, fields, test_data)
  validation_data = []
  get_file_data(validation_file, fields, validation_data)

  ce_results = [['0' for i in range(5)] for i in range(4)]
  ent_results = [['0' for i in range(5)] for i in range(4)]
  tab_labels = ['Confidence Level', 'Accuracy\t', 'Number Nodes\t', 'Tree Depth\t']
  for i in range(4):
    ce_results[i][0] = tab_labels[i]
    ent_results[i][0] = tab_labels[i]
  for i in range(1,4):
    ce_results[0][i+1] = str(chi_sq_vals[i])
    ent_results[0][i+1] = str(chi_sq_vals[i])
  print ('Using Classification Error as Criterion:')
  for i in range(1,5):
    cl = ce_results[0][i]
    print ('Confidence Level ',cl,'%')
    tree_root = get_tree(training_data, cl, False)
    test_tree(tree_root, ce_results, i, test_data)
  print ('\nUsing Entropy as Criterion:')
  for i in range(1,5):
    cl = ent_results[0][i]
    print ('Confidence Level ',cl,'%')
    tree_root = get_tree(training_data, cl, True)
    test_tree(tree_root, ent_results, i, test_data)
  print('\n')
  print ('Criterion = Classification Error')
  for row in ce_results:
    print ('\t'.join(row))
  print ('\nCriterion = Entropy')
  for row in ent_results:
    print ('\t'.join(row))

  # handle validation data
  val_tree = get_tree(training_data, 0, True)
  predictions = classify_instance_list(val_tree, target_attrib, validation_data)
  if output_path:
    output_csv(output_path, predictions)

