import math, pickle
import numpy, operator
#Stores the test and training data as lists of lists
#Training data[User][Movie Ratings]
#Test data[User][Movie Ratings]
def parse_data():

# TRAINING FILE
	train_data = []
	test_data = []
	test_users = []
	test_files = ["test5.txt", "test10.txt", "test20.txt"]
	train_file = "train.txt"

	
	with open(train_file, "r") as f:
		train = f.read()

	users = train.split('\n')
	users.pop()

	for user in users:
		ratings = user.split('\t')
		for rating in ratings:
			rating = int(rating)
		train_data.append(ratings)

	#train_data = numpy.transpose(train_data)
	train_file = train_file.replace(".txt", ".data")
	with open(train_file, "wb") as f:
			pickle.dump(train_data, f)

# TEST FILE
	user_ratings = []



	for test_file in test_files:

		user_ratings = [[]]*200

		movie_ratings = [0]*1000

		with open(test_file, "r") as f:
			test = f.read()
		lines = test.split('\n')
		lines.pop()
		base_address = int(lines[0].split(' ')[0])
		prev_user = 0

		for line in lines:

			ratings = line.split(' ')

			if ratings[0] is not '':

				this_user = int(ratings[0])

				if prev_user is not this_user:
					prev_user = this_user
					movie_ratings = [0]*1000

				this_movie = int(ratings[1])
				this_movie_rating = int(ratings[2])
				movie_ratings[this_movie - 1] = this_movie_rating
				user_ratings[this_user - base_address] = movie_ratings	
		test_file = test_file.replace(".txt", ".data")
		with open(test_file, "wb") as f:
			pickle.dump(user_ratings, f)

# Calculate cosine similarity between all users for each user

def cosine_similarity(test=5):
	numpy.seterr(divide='ignore', invalid='ignore')

	train_file = "train.data"
	if test is 5:
		test_file = "test5.data"
		base_address = 201
	if test is 10:
		test_file = "test10.data"
		base_address = 301
	if test is 20:
		test_file = "test20.data"
		base_address = 401

	with open(train_file, "rb") as f:
		training_users = pickle.load(f)

	with open(test_file, "rb") as f:
		test_users = pickle.load(f)
		similarities = []
	# Calculate per test user
	test_user_id = base_address - 1
	for test_user in test_users:

		test_user_id = test_user_id + 1
		training_user_id = 0
		max_sim = 0
		max_sim_user_id = 0
		if not test_user:
			continue
		# Calculare similarity between this test and each training user
		for training_user in training_users:
			training_user_id = training_user_id + 1

			training_user = list(map(int, training_user))
			test_user = list(map(int, test_user))
			if(numpy.linalg.norm(training_user) * numpy.linalg.norm(test_user)) is not 0:
				this_sim = numpy.dot(training_user, test_user) / (numpy.linalg.norm(training_user) * numpy.linalg.norm(test_user))
			else:
				print("DIVIDE BY ZERO")
				this_sim = 0.0

			similarities.append((this_sim,training_user_id))
		# Sort similarity from high to low, preserving id of training user
		sorted_sims = sorted(similarities, key = operator.itemgetter(0), reverse = True)

		movie_id = 0
		#Find movies with no rating
		for rating in test_user:
			movie_id = movie_id + 1

			if rating is 0:
				#Traverse from most similar to least, find first person with a valid rating
				for similar_user in sorted_sims:
					sim_user_id = similar_user[1]
					movie_training_rating = int(training_users[sim_user_id - 1][movie_id - 1])

					if movie_training_rating is not 0:
						test_users[test_user_id - base_address][movie_id - 1] = movie_training_rating
						break
	print(test_users)
	print(len(test_users))




if __name__ == '__main__':
	parse_data()
	cosine_similarity()


