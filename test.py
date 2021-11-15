""" from mnist import MNIST

mndata = MNIST('samples')

images, labels = mndata.load_training()

print(labels[1]) """

a = ["Covid", "Other"]
print(a.index("Covid"))