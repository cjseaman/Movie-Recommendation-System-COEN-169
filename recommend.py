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
		train_data.append(list(map(int, ratings)))

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

		for line in lines:

			ratings = line.split(' ')

			#if ratings[0] is not '':

			this_user = int(ratings[0])

			this_movie = int(ratings[1])
			this_movie_rating = int(ratings[2])

			if prev_user != this_user:
				prev_user = this_user
				movie_ratings = [0]*1000

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

		if not test_user:
			continue

		# Calculare similarity between this test user and each training user
		for training_user in training_users:
			training_user_id = training_user_id + 1

			training_user = list(map(int, training_user))
			test_user = list(map(int, test_user))

			if(numpy.linalg.norm(training_user) * numpy.linalg.norm(test_user)) != 0:
				this_sim = numpy.dot(training_user, test_user) / numpy.linalg.norm(training_user) * numpy.linalg.norm(test_user)
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

			if rating == 0:
				predicted_rating = 0
				sim_value = 0
				total_sim_value = 0
				#Traverse from most similar to least, find first person with a valid rating
				for similar_user in sorted_sims:
					sim_user_id = similar_user[1]
					sim_value = similar_user[0]
					movie_training_rating = int(training_users[sim_user_id - 1][movie_id - 1])

					if movie_training_rating != 0:
						total_sim_value += sim_value
						predicted_rating += movie_training_rating * sim_value

				#If no valid ratings exist, use average of user's ratings
				if predicted_rating == 0:
					n = 0
					for rating in test_user:
						if rating != 0:
							predicted_rating += rating
							n += 1
					predicted_rating = int(round(predicted_rating / n))
					test_users[test_user_id - base_address][movie_id - 1] = predicted_rating
				else:
					#print("User:", test_user_id)
					#print("Aggregated predicted rating:", predicted_rating)
					#print("Total similarity:", total_sim_value)
					predicted_rating = int(round(predicted_rating / total_sim_value))
					#print("Calculated predicted_rating:", predicted_rating)
					if predicted_rating > 5 or predicted_rating < 1:
						print("INVALID PREDICTION:", predicted_rating)
						return
					test_users[test_user_id - base_address][movie_id - 1] = predicted_rating


	test_file_output = test_file.replace(".data", "_out.txt")
	test_file_input = test_file.replace(".data", ".txt")

	with open(test_file_input, "r") as f:
		reference_test_file = f.read()

	reference_lines = reference_test_file.split("\n")
	reference_lines.pop()
	line_number = 0
	output_lines = []

	for line in reference_lines:
		line_number += 1
		line_items = line.split(' ')
		if int(line_items[2]) is 0:
			output_lines.append(reference_lines[line_number - 1].replace(" 0", ' ' + str( test_users[ int(line_items[0]) - base_address ][ int(line_items[1]) - 1 ])))

	output = "\n".join(output_lines) + '\n'

	with open(test_file_output, "w+") as f:
		f.write(output)


def pearson_correlation(test=5):

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

	# Save average ratings
	test_users_avg_ratings = []
	training_users_avg_ratings = []

	# Calculate total average ratings of each user
	for test_user in test_users:
		n = 0
		average_rating = 0
		for rating in test_user:
			if rating != 0:
				average_rating += rating
				n += 1
		average_rating = average_rating / n
		test_users_avg_ratings.append(average_rating)

	for training_user in training_users:
		n = 0
		average_rating = 0
		for rating in training_user:
			if rating != 0:
				average_rating += rating
				n += 1
		average_rating = average_rating / n
		training_users_avg_ratings.append(average_rating)

	#Create list of adjusted ratings

	training_users_adjusted = []
	training_user_id = 0

	for avg in training_users_avg_ratings:
		training_user_id += 1
		adjusted_ratings = []

		for rating in training_users[training_user_id - 1]:
			adjusted_ratings.append(rating - avg)

		training_users_adjusted.append(adjusted_ratings)
	
	test_users_adjusted = []
	test_user_id = 0

	for avg in test_users_avg_ratings:
		test_user_id += 1
		adjusted_ratings = []

		for rating in test_users[test_user_id - 1]:
			adjusted_ratings.append(rating - avg)

		test_users_adjusted.append(adjusted_ratings)

	#Calculate average adjusted ratings
	training_users_adjusted_avg_ratings = []

	training_user_id = 0
	for training_user in training_users_adjusted:
		training_user_id += 1
		n = 0
		average_rating = 0
		training_rating_id = 0
		for rating in training_user:
			training_rating_id += 1
			if training_users[training_user_id - 1][training_rating_id - 1] != 0:
				average_rating += rating
				n += 1
		average_rating = average_rating / n
		training_users_adjusted_avg_ratings.append(average_rating)

	similarities = []

	# Calculate per test user
	test_user_id = base_address - 1
	for test_user in test_users_adjusted:

		test_user_id = test_user_id + 1
		training_user_id = 0

		if not test_user:
			continue

		# Calculare similarity between this test user and each training user
		for training_user in training_users_adjusted:
			training_user_id = training_user_id + 1

			if(numpy.linalg.norm(training_user) * numpy.linalg.norm(test_user)) != 0:
				this_sim = numpy.dot(training_user, test_user) / (numpy.linalg.norm(training_user) * numpy.linalg.norm(test_user))
			else:
				print("DIVIDE BY ZERO")
				this_sim = 0.0

			similarities.append((this_sim,training_user_id))
		# Sort similarity from high to low, preserving id of training user
		sorted_sims = sorted(similarities, key = operator.itemgetter(0), reverse = True)

		movie_id = 0
		#Find movies with no rating
		for rating in test_users[test_user_id - base_address]:
			movie_id = movie_id + 1

			if rating == 0:
				predicted_rating = 0
				sim_value = 0
				total_sim_value = 0

				#Traverse from most similar to least
				for similar_user in sorted_sims:

					sim_user_id = similar_user[1]
					sim_value = similar_user[0]

					if training_users[sim_user_id - 1][movie_id - 1] != 0:

						movie_training_rating = training_users_adjusted[sim_user_id - 1][movie_id - 1]# - training_users_adjusted_avg_ratings[sim_user_id - 1]
						total_sim_value += abs(sim_value)
						predicted_rating += movie_training_rating * sim_value
				
				if total_sim_value == 0:
					print("Using average")
					test_users[test_user_id - base_address][movie_id - 1] = int(round(predicted_rating))
				else:
					predicted_rating = predicted_rating / total_sim_value
					predicted_rating += training_users_avg_ratings[test_user_id - base_address]

					#print("Calculated predicted_rating:", predicted_rating)
					if predicted_rating < 1:
						#print("INVALID PREDICTION:", predicted_rating)
						predicted_rating = 1
					if predicted_rating > 5:
						#print("INVALID PREDICTION:", predicted_rating)
						predicted_rating = 5

					test_users[test_user_id - base_address][movie_id - 1] = int(round(predicted_rating))

	
	test_file_output = test_file.replace(".data", "_out.txt")
	test_file_input = test_file.replace(".data", ".txt")

	with open(test_file_input, "r") as f:
		reference_test_file = f.read()

	reference_lines = reference_test_file.split("\n")
	reference_lines.pop()
	line_number = 0
	output_lines = []

	for line in reference_lines:
		line_number += 1
		line_items = line.split(' ')
		if int(line_items[2]) is 0:
			output_lines.append(reference_lines[line_number - 1].replace(" 0", ' ' + str( test_users[ int(line_items[0]) - base_address ][ int(line_items[1]) - 1 ])))

	output = "\n".join(output_lines) + '\n'

	with open(test_file_output, "w+") as f:
		f.write(output)

if __name__ == '__main__':
	parse_data()
	tests = [5, 10, 20]
	for test in tests:
		print("Performing test", test)
		pearson_correlation(test)


