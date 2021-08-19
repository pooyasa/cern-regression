import pandas as pd
import numpy as np
from tqdm.notebook import tqdm

import seaborn as sns

import sklearn
import sklearn.metrics as metrics
from sklearn import preprocessing
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import SGDRegressor
from sklearn.svm import SVR

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt

class NN_Regressor(torch.nn.Module):
	def __init__(self):
		super(NN_Regressor, self).__init__()
		self.hid1 = torch.nn.Linear(16, 32)  # 16-(32-10)-1
		self.hid2 = torch.nn.Linear(32, 10)
		self.oupt = torch.nn.Linear(10, 1)
	def forward(self, x):
		x = torch.relu(self.hid1(x))
		x = torch.relu(self.hid2(x))
		x = self.oupt(x)  # no activation
		return x


class Optimized_NN_Regressor(torch.nn.Module):
	def __init__(self):
		super(Optimized_NN_Regressor, self).__init__()
		self.hid1 = torch.nn.Linear(16, 24)  # 16-(32-10)-1
		self.hid2 = torch.nn.Linear(24, 24)
		self.dropout = torch.nn.Dropout(p=0.1, inplace=False)
		self.oupt = torch.nn.Linear(24, 1);
	def forward(self, x):
		x = torch.relu(self.hid1(x))
		x = torch.relu(self.hid2(x))
		x = self.dropout(x)
		x = self.oupt(x)  # no activation
		return x



class RegressionDataset(Dataset):
	
	def __init__(self, X_data, y_data):
		self.X_data = X_data
		self.y_data = y_data
		
	def __getitem__(self, index):
		return self.X_data[index], self.y_data[index]
		
	def __len__ (self):
		return len(self.X_data)
	

def heatmap(X, Y, title = '', bin_num = 150, log_space=False, x_lim=False, y_lim=False, reverse = True):
	plt.rcParams["axes.grid"] = False
	x_lim = x_lim if x_lim else [X.min(), X.max()]
	y_lim = y_lim if y_lim else [Y.min(), Y.max()]
	if log_space:
		xedges = 10 ** np.linspace(np.log10(x_lim[0]), np.log10(x_lim[1]), bin_num + 1)
	else:
		xedges = np.linspace(x_lim[0], x_lim[1], bin_num + 1)
	
	yedges = np.linspace(y_lim[0],y_lim[1], bin_num + 1)
	H, xedges, yedges = np.histogram2d(X, Y, bins=(xedges, yedges))
	H = H[:,::-1] if reverse else H
	plt.axis('off')
	plt.title(title);
	plt.imshow(H.T, cmap=plt.get_cmap("turbo"));


def linear_regressor(X_train, y_train, X_test, y_test):
	lin_reg = LinearRegression()
	lin_reg.fit(X_train, y_train)
	# Calculate R2 for the regression problem
	R_2 = lin_reg.score(X_test, y_test)

	# Calculate MSE and MAE for the regression problem
	MAE = metrics.mean_absolute_error(y_test, lin_reg.predict(X_test))
	MSE = metrics.mean_squared_error(y_test, lin_reg.predict(X_test))

	# print(f"R_2 for the regression problem is: {R_2:.2f}")
	# print(f"MSE for the regression problem is: {MSE:.2f}")
	# print(f"MAE for the regression problem is: {MAE:.2f}")
	return lin_reg, [R_2, MAE, MSE]

def polynomial_regressor(X_train, y_train, X_test, y_test, degree=3):
	poly = PolynomialFeatures(degree=degree)
	X_	 = poly.fit_transform(X_train)
	X_Test = poly.fit_transform(X_test)

	clf = linear_model.LinearRegression()
	clf.fit(X_, y_train)
	# Calculate R2 for the regression problem
	R_2 = clf.score(X_Test, y_test)

	# Calculate MSE and MAE for the regression problem
	MAE = metrics.mean_absolute_error(y_test, clf.predict(X_Test))
	MSE = metrics.mean_squared_error(y_test, clf.predict(X_Test))

	# print(f"R_2 for the regression problem is: {R_2:.2f}")
	# print(f"MSE for the regression problem is: {MSE:.2f}")
	# print(f"MAE for the regression problem is: {MAE:.2f}")
	return clf, poly, [R_2, MAE, MSE]

def lasso_regressor(X_train, y_train, X_test, y_test, alpha=0.1):	
	lasso_regr = Lasso(alpha=alpha)
	lasso_regr.fit(X_train, y_train)
	# Calculate R2 for the regression problem
	R_2 = lasso_regr.score(X_test, y_test)

	# Calculate MSE and MAE for the regression problem
	MAE = metrics.mean_absolute_error(y_test, lasso_regr.predict(X_test))
	MSE = metrics.mean_squared_error(y_test, lasso_regr.predict(X_test))

	# print(f"R_2 for the regression problem is: {R_2:.2f}")
	# print(f"MSE for the regression problem is: {MSE:.2f}")
	# print(f"MAE for the regression problem is: {MAE:.2f}")
	return lasso_regr, [R_2, MAE, MSE]

def sgd_regressor(X_train, y_train, X_test, y_test, max_iter=100000, tol=0.01):
	sgdـregr = SGDRegressor(max_iter=max_iter, tol=tol)
	sgdـregr.fit(X_train, y_train)
	# Calculate R2 for the regression problem
	R_2 = sgdـregr.score(X_test, y_test)

	# Calculate MSE and MAE for the regression problem
	MAE = metrics.mean_absolute_error(y_test, sgdـregr.predict(X_test))
	MSE = metrics.mean_squared_error(y_test, sgdـregr.predict(X_test))

	# print(f"R_2 for the regression problem is: {R_2:.2f}")
	# print(f"MSE for the regression problem is: {MSE:.2f}")
	# print(f"MAE for the regression problem is: {MAE:.2f}")
	
	return sgdـregr, [R_2, MAE, MSE]


def svr_regressor(X_train, y_train, X_test, y_test,C=1.0, epsilon=0.2):

	regr = SVR(C=C, epsilon=epsilon)
	regr.fit(X_train, y_train)
	# Calculate R2 for the regression problem
	R_2 = regr.score(X_test, y_test)

	# Calculate MSE and MAE for the regression problem
	MAE = metrics.mean_absolute_error(y_test, regr.predict(X_test))
	MSE = metrics.mean_squared_error(y_test, regr.predict(X_test))

	# print(f"R_2 for the regression problem is: {R_2:.2f}")
	# print(f"MSE for the regression problem is: {MSE:.2f}")
	# print(f"MAE for the regression problem is: {MAE:.2f}")

	return regr, [R_2, MAE, MSE]


def random_forest_regressor(X_train, y_train, X_test, y_test,max_depth=20, random_state=0 ):
	#Random Forest
	regr = RandomForestRegressor(max_depth=max_depth, random_state=random_state)
	regr.fit(X_train, y_train)
	# Calculate R2 for the regression problem
	R_2 = regr.score(X_test, y_test)

	# Calculate MSE and MAE for the regression problem
	MAE = metrics.mean_absolute_error(y_test, regr.predict(X_test))
	MSE = metrics.mean_squared_error(y_test, regr.predict(X_test))

	# print(f"R_2 for the regression problem is: {R_2:.2f}")
	# print(f"MSE for the regression problem is: {MSE:.2f}")
	# print(f"MAE for the regression problem is: {MAE:.2f}")

	return regr, [R_2, MAE, MSE]


def neural_network_regressor(X_train, y_train, X_val, y_val, epochs = 150):
	train_dataset = RegressionDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float())
	val_dataset =   RegressionDataset(torch.from_numpy(X_val).float(), torch.from_numpy(y_val).float())

	BATCH_SIZE = 64
	LEARNING_RATE = 0.001

	train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
	val_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE)
	test_loader = DataLoader(dataset=val_dataset, batch_size=1)

	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	print(f"Running on: {device}")

	model = NN_Regressor()
	model.to(device)
	print(model)
	criterion = nn.MSELoss()
	optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
	print("Begin training.")

	loss_stats = {
		'train': [],
		"val": []
	}

	for e in tqdm(range(1, epochs+1)):
		
		# TRAINING
		train_epoch_loss = 0
		model.train()
		for X_train_batch, y_train_batch in train_loader:
			X_train_batch, y_train_batch = X_train_batch.to(device), y_train_batch.to(device)
			optimizer.zero_grad()
			
			y_train_pred = model(X_train_batch)
			
			train_loss = criterion(y_train_pred, y_train_batch.unsqueeze(1))
			
			train_loss.backward()
			optimizer.step()
			
			train_epoch_loss += train_loss.item()
			
			
		# VALIDATION	
		with torch.no_grad():
			
			val_epoch_loss = 0
			
			model.eval()
			for X_val_batch, y_val_batch in val_loader:
				X_val_batch, y_val_batch = X_val_batch.to(device), y_val_batch.to(device)
				
				y_val_pred = model(X_val_batch)
							
				val_loss = criterion(y_val_pred, y_val_batch.unsqueeze(1))
				
				val_epoch_loss += val_loss.item()
		loss_stats['train'].append(train_epoch_loss/len(train_loader))
		loss_stats['val'].append(val_epoch_loss/len(val_loader))							  
		
		# print(f'Epoch {e+0:03}: | Train Loss: {train_epoch_loss/len(train_loader):.5f} | Val Loss: {val_epoch_loss/len(val_loader):.5f}')
		train_val_loss_df = pd.DataFrame.from_dict(loss_stats).reset_index().melt(id_vars=['index']).rename(columns={"index":"epochs"})

	y_pred_list = []
	with torch.no_grad():
		model.eval()
		for X_batch, _ in test_loader:
			X_batch = X_batch.to(device)
			y_test_pred = model(X_batch)
			y_pred_list.append(y_test_pred.cpu().numpy())

	y_pred_list = [a.squeeze() for a in y_pred_list]
	y_pred_list = [a.squeeze().tolist() for a in y_pred_list]

	MAE = metrics.mean_absolute_error(y_val, y_pred_list)
	MSE = metrics.mean_squared_error(y_val, y_pred_list)
	R_2 = metrics.r2_score(y_val, y_pred_list)

	# print(f"R_2 for the regression problem is: {R_2:.2f}")
	# print(f"MSE for the regression problem is: {MSE:.2f}")
	# print(f"MAE for the regression problem is: {MAE:.2f}")
	return model.cpu(),train_val_loss_df , [R_2, MAE, MSE]

def calculate_performance(perf_array):
	mean = np.array(perf_array).mean(axis=0);
	std = np.array(perf_array).std(axis=0);
	R_2_mean = mean[0];
	R_2_std = std[0];
	MSE_mean = mean[1];
	MSE_std = std[1];
	MAE_mean = mean[2];
	MAE_std = std[2];	


	print(f"R_2: {R_2_mean:.2f} ±{R_2_std:.2f}")
	print(f"MSE: {MSE_mean:.2f} ±{MSE_std:.2f}")
	print(f"MAE: {MAE_mean:.2f} ±{MAE_std:.2f}")

	return


def optimized_neural_network_regressor(X_train, y_train, X_val, y_val, epochs = 150):
	train_dataset = RegressionDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float())
	val_dataset =   RegressionDataset(torch.from_numpy(X_val).float(), torch.from_numpy(y_val).float())

	BATCH_SIZE = 64
	LEARNING_RATE = 0.01

	train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
	val_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE)
	test_loader = DataLoader(dataset=val_dataset, batch_size=1)

	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	print(f"Running on: {device}")

	model = Optimized_NN_Regressor()
	model.to(device)
	print(model)
	criterion = nn.MSELoss()
	optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
	print("Begin training.")

	loss_stats = {
		'train': [],
		"val": []
	}

	for e in tqdm(range(1, epochs+1)):
		
		# TRAINING
		train_epoch_loss = 0
		model.train()
		for X_train_batch, y_train_batch in train_loader:
			X_train_batch, y_train_batch = X_train_batch.to(device), y_train_batch.to(device)
			optimizer.zero_grad()
			
			y_train_pred = model(X_train_batch)
			
			train_loss = criterion(y_train_pred, y_train_batch.unsqueeze(1))
			
			train_loss.backward()
			optimizer.step()
			
			train_epoch_loss += train_loss.item()
			
			
		# VALIDATION	
		with torch.no_grad():
			
			val_epoch_loss = 0
			
			model.eval()
			for X_val_batch, y_val_batch in val_loader:
				X_val_batch, y_val_batch = X_val_batch.to(device), y_val_batch.to(device)
				
				y_val_pred = model(X_val_batch)
							
				val_loss = criterion(y_val_pred, y_val_batch.unsqueeze(1))
				
				val_epoch_loss += val_loss.item()
		loss_stats['train'].append(train_epoch_loss/len(train_loader))
		loss_stats['val'].append(val_epoch_loss/len(val_loader))							  
		
		# print(f'Epoch {e+0:03}: | Train Loss: {train_epoch_loss/len(train_loader):.5f} | Val Loss: {val_epoch_loss/len(val_loader):.5f}')
		train_val_loss_df = pd.DataFrame.from_dict(loss_stats).reset_index().melt(id_vars=['index']).rename(columns={"index":"epochs"})

	y_pred_list = []
	with torch.no_grad():
		model.eval()
		for X_batch, _ in test_loader:
			X_batch = X_batch.to(device)
			y_test_pred = model(X_batch)
			y_pred_list.append(y_test_pred.cpu().numpy())

	y_pred_list = [a.squeeze() for a in y_pred_list]
	y_pred_list = [a.squeeze().tolist() for a in y_pred_list]

	MAE = metrics.mean_absolute_error(y_val, y_pred_list)
	MSE = metrics.mean_squared_error(y_val, y_pred_list)
	R_2 = metrics.r2_score(y_val, y_pred_list)

	# print(f"R_2 for the regression problem is: {R_2:.2f}")
	# print(f"MSE for the regression problem is: {MSE:.2f}")
	# print(f"MAE for the regression problem is: {MAE:.2f}")
	return model.cpu(),train_val_loss_df , [R_2, MAE, MSE]

