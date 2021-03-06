#Core Python, Pandas, and kaldi_io
import numpy as np
import pandas as pd
import string
from collections import Counter

#Scikit
from sklearn import manifold
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import pairwise_distances, average_precision_score
from sklearn.metrics import PrecisionRecallDisplay, precision_recall_curve
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.dummy import DummyClassifier
from scipy import stats
from scipy.spatial.distance import pdist

#Plotting
from matplotlib import pyplot as plt
import seaborn as sns


#Torch and utilities
import torch
import torch.optim as optim
from torch.utils.data import TensorDataset,DataLoader,random_split,ConcatDataset


'''Script containing Utilities to train and test awe models as well as evaluate AWEs'''


############################################## Metrics and Losses #########################################################


def accuracy(out, yb):
	'''Accuracy of Classification Model'''
	preds = torch.argmax(out, dim=1)
	return (preds == yb).float().mean()

def cos_distance(cos,x_1,x_2):
	'''Cosine Distance'''
	return (1- cos(x_1,x_2))/2

def cos_hinge_loss(word_embedding,same_word_embedding,diff_word_embedding,cos, dev):
	'''Cosine Hinge Loss'''
	m = 0.15
	lower_bound = torch.tensor(0.0).to(dev, non_blocking = True)
	a = torch.max(lower_bound,m + cos_distance(cos, word_embedding, same_word_embedding) - cos_distance(cos, word_embedding, diff_word_embedding))
	return torch.mean(a)

############################################### Training Loops ############################################################

def siamese_train_loop(net,num_epochs,train_dl,val_dl,optimizer,dev,save_path = "./Models/siamese_best_model.pth",verbose = True):
	'''Training Loop to train a Siamese Network to bring Embeddings of Same labels close, while pushing embeddings of different labels apart'''

	#Whether to save model every few epochs
	save_epochs = False


	hist = {}
	train_loss_list = []
	val_loss_list = []
	model_save_path = save_path
	best_val_loss = np.Inf

	#Cosine Similarity
	cos = nn.CosineSimilarity(dim=1, eps=1e-6)

	for epoch in range(0,num_epochs):
		if verbose:
				print('epoch %d '%(epoch))

		train_loss = 0
		net.train()
		for batch_idx, (train_data,train_labels) in enumerate(train_dl):

			#print(train_data.shape)
			#Move to GPU
			optimizer.zero_grad()
			train_data = train_data.to(dev, non_blocking=True)
			word = train_data[:,0,:]
			same_word = train_data[:,1,:]
			diff_word = train_data[:,2,:]

			word_embedding = net(word)
			same_word_embedding = net(same_word)
			diff_word_embedding = net(diff_word)

			loss = cos_hinge_loss(word_embedding,same_word_embedding,diff_word_embedding, cos, dev)
			loss.backward()
			optimizer.step()
			train_loss += loss.item()


		net.eval()
		with torch.no_grad():
			val_loss = 0
			for batch_idx, (val_data,val_labels) in enumerate(val_dl):

				val_data = val_data.to(dev, non_blocking=True)
				word = val_data[:,0,:]
				same_word = val_data[:,1,:]
				diff_word = val_data[:,2,:]

				word_embedding = net(word)
				same_word_embedding = net(same_word)
				diff_word_embedding = net(diff_word)

				val_loss += cos_hinge_loss(word_embedding,same_word_embedding,diff_word_embedding, cos, dev)

			if val_loss < best_val_loss:
				best_val_loss = val_loss
				print("Best val loss %.3f Saving Model..."%(val_loss/len(val_dl)))
				torch.save(net.state_dict(),model_save_path)


		if verbose:
			print("train loss: %.3f"%(train_loss/len(train_dl)))
			print("val loss: %.3f"%(val_loss/len(val_dl)))
		if epoch%5 == 0 and save_epochs:
			#path = save_path + "simple_awe_bs64_epoch_%d.pth"%(epoch)
			epoch_save_path = "/data/users/jmahapatra/models/" + "siamese_epoch_%d.pth"%(epoch)
			torch.save(net.state_dict(), epoch_save_path)

		train_loss_list.append(train_loss/len(train_dl))
		val_loss_list.append(val_loss/len(val_dl))

	#Save the last model
	last_epoch_save_path = model_save_path[:-4]+ "_last_epoch.pth"
	torch.save(net.state_dict(), last_epoch_save_path)

	hist['train_loss'] = train_loss_list
	hist['val_loss'] = val_loss_list
	print('Finished Training')
	return hist



def classifier_train_loop(net,num_epochs,train_dl,val_dl,optimizer,criterion,dev,save_path = "./Models/awe_best_model.pth",verbose = True):
	'''Training Loop for a classifier'''
	
	#Whether to save model every few epochs
	save_epochs = False


	hist = {}
	train_acc_list = []
	val_acc_list = []
	train_loss_list = []
	val_loss_list = []
	
	best_val_loss = np.Inf
	best_val_acc = np.NINF
	print('Starting Training')
	for epoch in range(0,num_epochs):  # loop over the dataset multiple times
		if verbose:
			print('epoch %d '%(epoch))

		train_loss = 0
		train_acc = 0
		net.train()
		for xb,yb in train_dl:
			
			#Move to GPU
			xb,yb = xb.to(dev, non_blocking=True),yb.to(dev, non_blocking=True)
			
			# get the inputs; data is a list of [inputs, labels]
			#inputs, labels = torch.tensor(inputs),torch.tensor(labels)
			#labels = labels.long()
			# zero the parameter gradients
			optimizer.zero_grad()

			# forward + backward + optimize
			outputs = net(xb)
			loss = criterion(outputs, yb.long())
			train_loss += loss
			train_acc += accuracy(outputs,yb)
			loss.backward()
			optimizer.step()

			# print statistics
			running_loss = loss.item()
		net.eval()
		with torch.no_grad():
			val_loss = 0
			val_acc = 0
			for xb,yb in val_dl:
				
				#Move to GPU
				xb,yb = xb.to(dev, non_blocking=True),yb.to(dev, non_blocking=True)
				
				#get predictions
				y_pred = net(xb)
				
				val_loss += criterion(y_pred,yb.long())
				val_acc += accuracy(y_pred, yb.long())
			#val_loss = sum(criterion(net(xb), yb.long()) for xb, yb in val_dl)
			#val_acc = sum(accuracy(net(xb), yb.long()) for xb, yb in val_dl)
			if val_acc > best_val_acc:
				best_val_acc = val_acc
				#path = save_path + "awe_best_model.pth"
				if verbose:
					print("Best val acc. Saving model...")
				torch.save(net.state_dict(), save_path)

		
		
		if verbose:
			print("train loss: %.3f train acc: %.3f"%(train_loss/len(train_dl),train_acc/len(train_dl)))
			print("val loss: %.3f val acc: %.3f"%(val_loss/len(val_dl),val_acc/len(val_dl)))
		if epoch%5 == 0 and save_epochs:
			#path = save_path + "simple_awe_bs64_epoch_%d.pth"%(epoch)
			epoch_save_path = "/data/users/jmahapatra/models/" + "awe_bs64_epoch_%d.pth"%(epoch)
			torch.save(net.state_dict(), epoch_save_path)


		train_loss_list.append(train_loss.item()/len(train_dl))
		train_acc_list.append(train_acc.item()/len(train_dl))
		val_loss_list.append(val_loss.item()/len(val_dl))
		val_acc_list.append(val_acc.item()/len(val_dl))

	
	hist['train_acc'] = train_acc_list
	hist['val_acc'] = val_acc_list
	hist['train_loss'] = train_loss_list
	hist['val_loss'] = val_loss_list
	print('Finished Training')
	return hist


################################################################## Model Testing ###########################################################

def test_classifier(net,test_dl,dev):
	'''Test Classification Accuracy'''
	test_acc = 0
	if dev.type == 'cuda':
		for xb,yb in test_dl:
			#Move to GPU
			xb,yb = xb.to(dev, non_blocking=True),yb.to(dev, non_blocking=True)
			y_pred = net(xb)
			test_acc += accuracy(y_pred,yb.long())
		test_acc = test_acc/len(test_dl)
	elif dev.type == 'cpu':
		for xb,yb in test_dl:
			y_pred = net(x_test)
			test_acc += accuracy(y_pred,yb.long())
		test_acc = test_acc/len(test_dl)

	print("Test Accuracy of best model is %f"%(test_acc))

	if dev.type == 'cuda':
		test_acc = test_acc.detach().cpu().numpy()
	else:
		test_acc = test_acc.detach().numpy()
	return test_acc


def test_siamese_model(net,test_dl, dev):
	test_loss = 0
	net.eval()

	#Cosine Similarity
	cos = nn.CosineSimilarity(dim=1, eps=1e-6)

	with torch.no_grad():
		for i,(test_data,test_labels) in enumerate(test_dl):

			#print(i)

			#show_cuda_memory()
			#if dev.type == 'cuda' and not test_data.is_cuda:
			#    test_data = test_data.to(dev, non_blocking=True)

			word = test_data[:,0,:].to(dev)
			same_word = test_data[:,1,:].to(dev)
			diff_word = test_data[:,2,:].to(dev)

			word_embedding = net(word)
			same_word_embedding = net(same_word)
			diff_word_embedding = net(diff_word)

			test_data.to('cpu')

			test_loss += cos_hinge_loss(word_embedding,same_word_embedding,diff_word_embedding, cos, dev)
			#show_cuda_memory()
	final_test_loss = test_loss/len(test_dl)
	print("Test Loss is %.3f"%(final_test_loss))
	return final_test_loss



def baseline(train_ds, test_ds):
	'''Baseline Classification Accuracy'''

	x_train, y_train = train_ds.inputs.numpy(), train_ds.labels.numpy()
	x_test, y_test = test_ds.inputs.numpy(), test_ds.labels.numpy()

	#Create Dummy classifier
	dummy_clf = DummyClassifier(strategy="most_frequent")

	#Fit dummy clf
	dummy_clf.fit(x_train, y_train)

	#Return mean accuracy on test
	return dummy_clf.score(x_test, y_test)



	############################################################### Evaluate AWEs ##########################################################


#Evaluate Embeddings
def evaluate_embeddings(embeddings, labels, curve = False):
	
	'''Evaluate AWEs based on Same-Different Task
	Same-Different Task is to classify if a pair of embeddings belong to the same label,
	given their distance. The metric reported is average precision.

	Arguments:
		embeddings (Numpy Array): An Array of N x embedding_dimension for embeddings of N word instances
		labels (Numpy Array): Labels of the N word instances
		curve (Boolean, default -> False) : Whether the whole precision-recall curve is to be returned

	Returns:
		average_precision (Float) : Average Precision calculated on the Same-Different Task
		curve_points (Tuple of List) : A tuple containing precision and recall points for the Precision-Recall Curve
	
	'''

	#Calculate pairwise cosine distance
	distances = pairwise_distances(embeddings, metric='cosine')
	#Calculate pairwise cosine similarity
	similarity = pairwise_kernels(embeddings, metric = 'cosine')
	
	
	
	#Create labels of whether the words are same or not
	eval_labels = (labels[:,None]==labels).astype(float)
	
	
	
	#Remove the diagonal elements (word pairs with themselves)
	mask = np.array(np.tril(np.ones((similarity.shape[0],similarity.shape[0]), dtype= int),-1),dtype = bool)
	similarity = similarity[mask]
	distances = distances[mask]
	eval_labels = eval_labels[mask]
		
	#flatten the pairwise arrays
	distances = np.ravel(distances)
	similarity = np.ravel(similarity)
	#Flatten the labels
	eval_labels = np.ravel(eval_labels)
	
	num_positive = sum(eval_labels==1)
	num_negative = eval_labels.shape[0]-num_positive
	print('The number of positive examples %d and negative examples %d'%(num_positive,num_negative))
	#Calculate the Average Precision
	average_precision = average_precision_score(eval_labels,similarity)
	if curve:
		precision, recall, _ = precision_recall_curve(eval_labels,similarity)
		curve_points = (precision, recall)
	else:
		curve_points = (None, None)

	return average_precision, curve_points



def evaluate_model(net,test_dl, dev, num_examples = np.Inf, curve_path = None):
	embeddings = None
	labels = None
	for i,(xb,yb) in enumerate(test_dl):
		#If device is GPU move features to GPU
		if dev.type == 'cuda' and not xb.is_cuda:
			xb = xb.to(dev, non_blocking=True)
			
		#Get the embeddings
		batch_embeddings = net.give_embeddings(xb, dev)
		#batch_embeddings = net(xb).cpu().detach().numpy()
		
		#Add to the main embeddings
		if embeddings is not None:
			embeddings = np.vstack((embeddings, batch_embeddings))
			labels = np.concatenate((labels,yb),axis=0)
		else:
			embeddings = batch_embeddings
			labels = yb

		if embeddings.shape[0]>num_examples:
			break

	if num_examples < np.Inf:
		embeddings = embeddings[:num_examples]
		labels = labels[:num_examples]

	print("Size of labels %d"%(labels.shape[0]))
	print("Number of unique words %d"%(np.unique(labels).shape[0]))

	avg_p, curve_points = evaluate_embeddings(embeddings, labels, curve = True)


	if curve_path is not None:
		#Save the precision recall curve
		print("saving precision recall curve")
		precision, recall = curve_points[0],curve_points[1]
		print(curve_path)
		disp = PrecisionRecallDisplay(precision=precision, recall=recall)
		disp.plot()
		plt.savefig(curve_path)
		plt.close()

	
	return avg_p



def evaluate_model_paper(net,evaluate_dl, dev,show_plot = True):
	'''Function used by Kamper et al. for Same-Different Task'''

	#Internal function
	def generate_matches_array(labels):
		"""
		Return an array of bool in the same order as the distances from
		`scipy.spatial.distance.pdist` indicating whether a distance is for
		matching or non-matching labels.
		"""
		N = len(labels)
		matches = np.zeros(int(N*(N - 1)/2), dtype=np.bool)

		# For every distance, mark whether it is a true match or not
		cur_matches_i = 0
		for n in range(N):
			cur_label = labels[n]
			matches[cur_matches_i:cur_matches_i + (N - n) - 1] = np.asarray(labels[n + 1:]) == cur_label
			cur_matches_i += N - n - 1

		return matches
	
	embeddings = None
	labels = None
	for i, (xb,yb) in enumerate(evaluate_dl):
		#If device is GPU move features to GPU
		if dev.type == 'cuda' and not xb.is_cuda:
			xb = xb.to(dev, non_blocking=True)
			
		#Get the embeddings
		batch_embeddings = net.give_embeddings(xb, dev)
		#batch_embeddings = net(xb).cpu().detach().numpy()
		
		#Add to the main embeddings
		if embeddings is not None:
			embeddings = np.vstack((embeddings, batch_embeddings))
			labels = np.concatenate((labels,yb),axis=0)
		else:
			embeddings = batch_embeddings
			labels = yb

		if i==2:
			break


	print('Finished getting %d embeddings'%(embeddings.shape[0]))
	#Calculate pairwise cosine distance
	distances = pdist(embeddings,metric = 'cosine')
	#Getting matches array
	matches = generate_matches_array(labels)
	
	
	pos_distances =  distances[matches == True]
	neg_distances = distances[matches == False]
	
	distances = np.concatenate([pos_distances, neg_distances])
	matches = np.concatenate([np.ones(len(pos_distances)), np.zeros(len(neg_distances))])
	
	
	
	sorted_i = np.argsort(distances)
	distances = distances[sorted_i]
	matches = matches[sorted_i]
	
	
	# Calculate precision
	precision = np.cumsum(matches)/np.arange(1, len(matches) + 1)
	
	
	# Calculate average precision: the multiplication with matches and division
	# by the number of positive examples is to not count precisions at the same
	# recall point multiple times.
	average_precision = np.sum(precision * matches) / len(pos_distances)
	print('Average Precision is %f'%(average_precision))
	# Calculate recall
	recall = np.cumsum(matches)/len(pos_distances)
	
	# More than one precision can be at a single recall point, take the max one
	for n in range(len(recall) - 2, -1, -1):
		precision[n] = max(precision[n], precision[n + 1])
		
	# Calculate precision-recall breakeven
	prb_i = np.argmin(np.abs(recall - precision))
	prb = (recall[prb_i] + precision[prb_i])/2.
	
	if show_plot:
		import matplotlib.pyplot as plt
		plt.plot(recall, precision)
		plt.xlabel("Recall")
		plt.ylabel("Precision")






def evaluate_siamese_model(net,test_dl, dev, num_examples = 11000):
	
	embeddings = None
	labels = None
	
	with torch.no_grad():
		for i, (test_data,test_labels) in enumerate(test_dl):
			#If device is GPU move features to GPU
			if dev.type == 'cuda' and not test_data.is_cuda:
				test_data = test_data.to(dev, non_blocking=True)

			word = test_data[:,0,:]
			same_word = test_data[:,1,:]
			diff_word = test_data[:,2,:]

			word_embedding = net(word).cpu().detach().numpy()
			same_word_embedding = net(same_word).cpu().detach().numpy()
			diff_word_embedding = net(diff_word).cpu().detach().numpy()

			word_labels = test_labels[:,0]
			same_word_labels = test_labels[:,0]
			diff_word_labels = test_labels[:,1]

			#Add to the main embeddings
			if embeddings is not None:
				embeddings = np.vstack((embeddings,word_embedding,same_word_embedding,diff_word_embedding))
				labels = np.concatenate((labels,word_labels,same_word_labels,diff_word_labels),axis=0)
			else:
				embeddings = np.vstack((word_embedding,same_word_embedding,diff_word_embedding))
				labels = np.concatenate((word_labels,same_word_labels,diff_word_labels),axis=0)
			
			if embeddings.shape[0] > num_examples:
				break
	
	avg_p, precision, recall = evaluate_embeddings(embeddings, labels)
	return avg_p





	

###################################################################### Misc ################################################################3


def clean_mapping(yb, clean_word_to_num, noisy_num_to_word):
	'''Map given labels to mapping of the clean dataset'''
	mapped_yb = [clean_word_to_num[noisy_num_to_word[yb[i].item()]] for i in range(yb.shape[0])]
	mapped_yb = torch.tensor(mapped_yb)
	
	return mapped_yb
	

def plot_learning_curves(hist,name = 'learning_curves.png', show = True):
	'''Plot Learning Curves using model training history'''
	
	num_epochs = len(hist['train_loss'])
	fig, axs = plt.subplots(2, 1, figsize = (9,9))

	axs[0].plot(np.arange(0,num_epochs,1), hist['train_loss'], label = "train_acc")
	axs[0].plot(np.arange(0,num_epochs,1),hist['val_loss'], label = "val_acc")
	axs[0].legend(loc = "best")
	axs[0].set_xlabel("Epochs")
	axs[0].set_ylabel("Loss")
	axs[0].set_title("Learning Curves (Loss)")

	if 'train_acc' in hist.keys():
		axs[1].plot(np.arange(0,num_epochs,1),list(map(lambda x: x*100,hist['train_acc'])), label = "train_acc")
		axs[1].plot(np.arange(0,num_epochs,1),list(map(lambda x: x*100,hist['val_acc'])), label = "val_acc")
		axs[1].legend(loc = "best")
		axs[1].set_xlabel("Epochs")
		axs[1].set_ylabel("Accuracy (in %)")
		axs[1].set_title("Learning Curves (Accuracy)")




	fig.tight_layout(pad = 2.0)
	plt.savefig(name)
	if show:
		plt.show()
	plt.close()


def run_data_study(splits,num_epochs,train_ds,val_dl,dev):
	'''Study Model Performance by varying data size'''
	lengths = [int(len(train_ds)*(1/splits)) for  i in range(splits)]
	
	if sum(lengths) != len(train_ds):
		lengths.append(len(train_ds) - sum(lengths))
		
	
	iteration_datasets = list(random_split(train_ds, lengths))
	
	#Remove the last element
	lengths = lengths[:-1]
	iteration_datasets = iteration_datasets[:-1]
	iteration_best_train_loss = []
	iteration_best_val_loss = []
	iteration_best_train_acc = []
	iteration_best_val_acc = []
	
	for iteration in range(splits):
		print('Iteration %d'%(iteration))
		iteration_ds = ConcatDataset(iteration_datasets[:iteration+1])
		iteration_dl = DataLoader(iteration_ds, batch_size=bs, pin_memory = True, drop_last = True)
		
		print('Training set having %d examples'%(len(iteration_ds)))
		#Create model
		net = SimpleNet()
		net = net.float()
		net.to(dev)
		
		hist = train_model(net,num_epochs,iteration_dl,val_dl,save_path="./data_study/", verbose = False)
		
		#Delete model
		del net
		
		best_train_loss = min(hist['train_loss'])
		best_val_loss = min(hist['val_loss'])
		best_train_acc = max(hist['train_acc'])
		best_val_acc = max(hist['val_acc'])
		
		iteration_best_train_loss.append(best_train_loss)
		iteration_best_val_loss.append(best_val_loss)
		iteration_best_train_acc.append(best_train_acc)
		iteration_best_val_acc.append(best_val_acc)
	
	#Plot the data study curves
	fig, axs = plt.subplots(2, 1, figsize = (9,9))

	axs[0].plot(np.arange(0,splits,1),iteration_best_train_loss, label = "train_acc")
	axs[0].plot(np.arange(0,splits,1),iteration_best_val_loss, label = "val_acc")
	axs[0].legend(loc = "best")
	axs[0].set_xlabel("Splits")
	axs[0].set_ylabel("Best Loss")
	axs[0].set_title("Data Splits vs Loss")

	axs[1].plot(np.arange(1,splits+1,1)*100/(splits),list(map(lambda x: x*100,iteration_best_train_acc)), label = "train_acc")
	axs[1].plot(np.arange(1,splits+1,1)*100/(splits),list(map(lambda x: x*100,iteration_best_val_acc)), label = "val_acc")
	axs[1].legend(loc = "best")
	axs[1].set_xlabel("Data Used (%)")
	axs[1].set_ylabel("Best Accuracy (in %)")
	axs[1].set_title("Data Splits vs Accuracy")



	name = 'data_study.png'
	fig.tight_layout(pad = 2.0)
	plt.savefig(name)
	plt.show()
		
		
	
			
			
			
			