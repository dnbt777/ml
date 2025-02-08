import pickle

def load_model_params(filename):
  with open(filename, 'rb') as file:
    obj = pickle.load(file)
  return obj

def save_model_params(params, filename="params.pkl"):
  with open(filename, 'wb') as file:
    pickle.dump(params, file)