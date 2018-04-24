import os
import csv

def categorization(labels):
    # This function categorize the labels into three categorise: cat, dog and others
    # Input is list of labels, e.g. labels returned by Google Cloud. Each label is a single string possibly containing spaces.
    # Output is also a list of categorized labels with value "cat" "dog", and "others". '''
    catFile = open(os.path.dirname(__file__)+"\cat.txt",'r')
    catSyn = csv.reader(catFile)
    dogFile = open(os.path.dirname(__file__)+"\dog.txt",'r')
    dogSyn = csv.reader(dogFile)  
    
    cateLabels = []
    for label in labels:
        cateLabels.append(_cateSingle(label, catSyn, dogSyn))
    return cateLabels
    

                    
          
        
    
def _cateSingle(label, catSyn, dogSyn):
    # This is a private function which categorize a single label
    for row in catSyn:
        for item in row:            
            if item.lower() in label.lower():
                # Depending on the specific format, we may also want to remove the spaces using str.replace()
                return "cat"
    for row in dogSyn:
        for item in row:
            if item.lower() in label.lower():
                return "dog"
    
    return "others"
                    
    