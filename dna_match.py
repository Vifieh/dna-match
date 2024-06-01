import csv
import time

import numpy as np
from sklearn.neighbors import KDTree


class Allele:
    def __init__(self, allele1, allele2):
        self.allele1 = allele1
        self.allele2 = allele2


class UserLocus:
    def __init__(self, locus_name, allele):
        self.locus_name = locus_name
        self.allele = allele


# Define the list of loci
loci = [
    'D3S1358', 'TH01', 'D21S11', 'D18S51', 'SE33', 'D5S818', 'D13S317', 'D7S820',
    'D16S539', 'CSF1PO', 'D22S1045', 'vWA', 'D8S1179', 'TPOX', 'D2S1338', 'D19S433',
    'FGA', 'D10S1248', 'D12S391', 'D1S1656'
]
# Define the number of users and loci
num_users = 1000
num_loci = len(loci)

# Define the range of repeat numbers for each locus
min_repeat = 5
max_repeat = 20

def calculate_percentage_match(child_data, dataset_entry):
    matches = 0
    for i in range(num_loci):
        child_allele_1 = child_data[0][1][i].allele.allele1
        child_allele_2 = child_data[0][1][i].allele.allele2
        dataset_allele_1 = dataset_entry[1][i].allele.allele1
        dataset_allele_2 = dataset_entry[1][i].allele.allele2
        if (child_allele_1 == dataset_allele_1 or child_allele_2 == dataset_allele_2) or (
                child_allele_1 == dataset_allele_2 or child_allele_2 == dataset_allele_1):
            matches += 1
    percentage_match = (matches / num_loci) * 100
    return percentage_match


def is_user_unique(dataset, user):
    for existing_user in dataset:
        if all(existing_user[1][i].allele.allele1 == user[1][i].allele.allele1 and
               existing_user[1][i].allele.allele2 == user[1][i].allele.allele2 for i in range(len(loci))):
            return False
    return True


# Read the dataset from CSV
filename = f"simulated_dataset_{num_users}.csv"
dataset = []
with open(filename, mode='r') as file:
    reader = csv.reader(file)
    header = next(reader)  # Skip the header row
    for row in reader:
        user_id = row[0]
        user_loci = []
        for i in range(1, len(header), 1):
            locus_name = header[i]
            allele1, allele2 = map(int, row[i].split(','))
            allele = Allele(allele1, allele2)
            user_locus = UserLocus(locus_name, allele)
            user_loci.append(user_locus)
        dataset.append((user_id, user_loci))

# Read the child data from CSV
child_data_filename = "child_dataset.csv"
child_data = []
with open(child_data_filename, mode='r') as file:
    reader = csv.reader(file)
    header = next(reader)  # Skip the header row
    for row in reader:
        child_id = row[0]
        user_loci = []
        for i in range(1, len(header), 1):
            locus_name = header[i]
            allele1, allele2 = map(int, row[i].split(','))
            allele = Allele(allele1, allele2)
            user_locus = UserLocus(locus_name, allele)
            user_loci.append(user_locus)
        child_data.append((child_id, user_loci))

# Create the feature matrix for the dataset
dataset_features = []
for user_data in dataset:
    feature_vector = []
    for locus in user_data[1]:
        feature_vector.extend([locus.allele.allele1, locus.allele.allele2])
    dataset_features.append(feature_vector)

# Create the KD-tree model
start_time = time.time()
kdtree_model = KDTree(dataset_features)
end_time = time.time()

# Calculate the elapsed time for KD-tree creation
elapsed_time_kdtree = end_time - start_time

# Reshape the child's feature vector for prediction
child_feature_vector = []
for locus in child_data[0][1]:
    child_feature_vector.extend([locus.allele.allele1, locus.allele.allele2])
child_feature_vector = np.array(child_feature_vector).reshape(1, -1)

# Find the k nearest neighbors using the KD-tree
k = 5  # Number of nearest neighbors to retrieve
start_time = time.time()
distances, indices = kdtree_model.query(child_feature_vector, k)
end_time = time.time()

# Calculate the elapsed time for finding nearest neighbors
elapsed_time_knn = end_time - start_time

# Retrieve the k nearest neighbors from the dataset
nearest_neighbors = [dataset[index] for index in indices[0]]

# Calculate the percentage match for each nearest neighbor
percentage_matches = [calculate_percentage_match(child_data, neighbor) for neighbor in nearest_neighbors]

# Print the nearest neighbors and their percentage matches
for i in range(k):
    print(f"Nearest Neighbor {i + 1}:")
    print(f"User ID: {nearest_neighbors[i][0]}")
    print(f"Percentage Match: {percentage_matches[i]}%")
    print("------")

# Print the elapsed times
print("Elapsed time for KD-tree creation:", elapsed_time_kdtree, "seconds")
print("Elapsed time for finding nearest neighbors:", elapsed_time_knn, "seconds")
