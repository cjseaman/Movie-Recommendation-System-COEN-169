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

	for user in train_data:
		print(user[776])


	#train_data = numpy.transpose(train_data)
	train_file = train_file.replace(".txt", ".data")
	with open(train_file, "wb") as f:
			pickle.dump(train_data, f)

# TEST FILE
	user_ratings = []



	for test_file in test_files:

		user_ratings = [[]]*100

		movie_ratings = [0]*1000

		with open(test_file, "r") as f:
			test = f.read()
		lines = test.split('\n')
		lines.pop()
		base_address = int(lines[0].split(' ')[0])
		prev_user = 0
		#print("test lines:", len(lines))
		for line in lines:

			ratings = line.split(' ')

			if ratings[0] is not '':

				this_user = int(ratings[0])

				if prev_user is not this_user:
					prev_user = this_user
					movie_ratings = [0]*1000

				this_movie = int(ratings[1])
				#print(ratings)
				this_movie_rating = int(ratings[2])
				#print("user index=", this_user - base_address)
				#print("movie index=", this_movie - 1)
				movie_ratings[this_movie - 1] = this_movie_rating
				#print("movie_ratings=", movie_ratings)
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
		#print('test_user',test_user_id, ':')
		#print(test_user)

		# Calculare similarity between this test user and each training user
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

				#If no valid ratings exist, just use 3
				if movie_training_rating is 0:
					test_users[test_user_id - base_address][movie_id - 1] = 3




	test_file_output = test_file.replace(".data", "_out.txt")
	test_file_input = test_file.replace(".data", ".txt")

	with open(test_file_input, "r") as f:
		reference_test_file = f.read()

	reference_lines = reference_test_file.split("\n")
	reference_lines.pop()
	line_number = 0
	output_lines = []

	print("Trouble spot (should be zero):", test_users[4][776])

	for line in reference_lines:
		line_number += 1
		line_items = line.split(' ')
		if int(line_items[2]) is 0:
			#print("changing line", line_number, "to", reference_lines[line_number - 1].replace(" 0", ' ' + str( test_users[ int(line_items[0]) - base_address ][ int(line_items[1]) - 1 ])))
			output_lines.append(reference_lines[line_number - 1].replace(" 0", ' ' + str( test_users[ int(line_items[0]) - base_address ][ int(line_items[1]) - 1 ])))

	output = "\n".join(output_lines) + '\n'

	with open(test_file_output, "w+") as f:
		f.write(output)



if __name__ == '__main__':
	parse_data()
	cosine_similarity(5)
	cosine_similarity(10)
	cosine_similarity(20)


