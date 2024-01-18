import pickle
import sklearn
model = pickle.load(open('GPmodel.pkl', 'rb'))


print(model)