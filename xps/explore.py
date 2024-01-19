import pickle
import sklearn
model = pickle.load(open('GPmodel.pkl', 'rb'))


print(model)



desc = []
BE = []

#then iterate over the molecules to get the descriptors and BE
#BUG: chack with JOren for the disctionary
for mol in molAll:

    if ("C" in mol.symbols) == True:
        descMol = descriptor.calc(mol) #descriptor for each molecule

        if 'data' in descMol:
           desc_data = descMol['data'] #get the data from the descriptor object if exist

           for element in desc_data:
               desc.append(element) #append the data in the array desc

        for atom in mol:

            if atom.number == Z:
                BE.append(mol.todict()["GW_charged"][atom.index]) # pick up the binding energy of the target atomic charge and store it in the same order
                #BUG: check for missing data. Typically delta KS data are rather sparsed. Now input as nan. We might have to nan the descriptor too
                #Gaussian process doesn't allow nan

desc = np.array(desc)
BE = np.array(BE)

print(len(desc), len(BE))

X_train, X_test, y_train, y_test = train_test_split(desc, BE, test_size=0.8, random_state=42)
np.save('X_test.npy', X_test)
np.save('y_test.npy', y_test)

kernel = 1.0 * RBF(length_scale=1e-1, length_scale_bounds=(1e-2, 1e3)) + WhiteKernel(
    noise_level=1e-2, noise_level_bounds=(1e-10, 1e1)
)
gpr = GaussianProcessRegressor(kernel=kernel, alpha=0.0)
gpr.fit(X_train, y_train)