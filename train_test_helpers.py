#Core Python, Pandas, and kaldi_io
import numpy as np
import pandas as pd
import string
from collections import Counter
import kaldi_io
import pdb

#Scikit
from sklearn import manifold
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import pairwise_distances,average_precision_score
from sklearn.metrics import PrecisionRecallDisplay,precision_recall_curve
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.dummy import DummyClassifier
from scipy import stats
from scipy.spatial.distance import pdist

#Plotting
from matplotlib import pyplot as plt
import seaborn as sns


#Torch and utilities
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset,DataLoader,random_split,ConcatDataset



def accuracy(out, yb):
	preds = torch.argmax(out, dim=1)
	return (preds == yb).float().mean()

def cos_distance(cos,x_1,x_2):
    return (1- cos(x_1,x_2))/2

def cos_hinge_loss(word_embedding,same_word_embedding,diff_word_embedding,cos, dev):
    m = 0.15
    lower_bound = torch.tensor(0.0).to(dev, non_blocking = True)
    a = torch.max(lower_bound,m + cos_distance(cos, word_embedding, same_word_embedding) - cos_distance(cos, word_embedding, diff_word_embedding))
    return torch.mean(a)

def siamese_train_loop(net,num_epochs,train_dl,val_dl,optimizer,dev,save_path = "./Models/siamese_best_model.pth",verbose = True):


	#Whether to save model every few epochs
	save_epochs = False


	hist = {}
	train_loss_list = []
	val_loss_list = []
	model_save_path = save_path

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

		train_loss_list.append(train_loss.item()/len(train_dl))
		val_loss_list.append(val_loss.item()/len(val_dl))

	hist['train_loss'] = train_loss_list
	hist['val_loss'] = val_loss_list
	print('Finished Training')
	return hist



def train_loop(net,num_epochs,train_dl,val_dl,optimizer,criterion,dev,save_path = "./Models/awe_best_model.pth",verbose = True):
	
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

	#Calculate pairwise cosine distance
	distances = pairwise_distances(embeddings, metric='cosine')
	#Calculate pairwise cosine similarity
	similarity = pairwise_kernels(embeddings, metric = 'cosine')
	
	
	
	#Create labels of whether the words are same or not
	if torch.is_tensor(labels):
		labels = labels.detach().numpy()
		
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
	avg_p = average_precision_score(eval_labels,similarity)

	if curve_path is not None:
		#Save the precision recall curve
		print("saving precision recall curve")
		print(curve_path)
		precision, recall, _ = precision_recall_curve(eval_labels,similarity)
		disp = PrecisionRecallDisplay(precision=precision, recall=recall)
		disp.plot()
		plt.savefig(curve_path)
		plt.close()

	#avg_p = average_precision_score(eval_labels,2-distances)
	#avg_p = average_precision_score(eval_labels,2-distances)
	#print('Average Precision is %f'%(avg_p))
	return avg_p



def evaluate_model_paper(net,evaluate_dl, dev,show_plot = True):

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

def test_model(net,test_dl,dev):
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


def baseline(train_ds, test_ds):

	x_train, y_train = train_ds.inputs.numpy(), train_ds.labels.numpy()
	x_test, y_test = test_ds.inputs.numpy(), test_ds.labels.numpy()

	#Create Dummy classifier
	dummy_clf = DummyClassifier(strategy="most_frequent")

	#Fit dummy clf
	dummy_clf.fit(x_train, y_train)

	#Return mean accuracy on test
	return dummy_clf.score(x_test, y_test)




def plot_learning_curves(hist,name = 'learning_curves.png', show = True):
	
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
		
		
	
			
			
			
			