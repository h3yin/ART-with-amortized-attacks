import sys
import tempfile
#sys.path.append('/home/h3yin/CS260_ML/project/adversarial-robustness-toolbox/')
sys.path.append('.')

from art.attacks.evasion import BoundaryAttack, FastGradientMethod, ZooAttack, HopSkipJump, SimBA
from art.estimators.classification.query_efficient_bb import QueryEfficientGradientEstimationClassifier
from art.attacks.evasion.projected_gradient_descent.projected_gradient_descent import ProjectedGradientDescent

from art.estimators.classification import TensorFlowV2Classifier
from art.utils import load_mnist
#from tests.utils import master_seed, get_image_classifier_kr

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Sequential, load_model, save_model
from tensorflow.python.keras.layers import deserialize, serialize
from tensorflow.python.keras.saving import saving_utils

from sklearn.metrics.pairwise import euclidean_distances
from sklearn.utils import shuffle

import numpy as np
import pickle
import time
import gc

#from keras.engine import _minimum_control_deps


class TensorFlowModel(Model):
    """
    Standard TensorFlow model for unit testing.
    """

    def __init__(self):
        super(TensorFlowModel, self).__init__()
        self.conv1 = Conv2D(filters=4, kernel_size=5, activation="relu")
        self.conv2 = Conv2D(filters=10, kernel_size=5, activation="relu")
        self.maxpool1 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding="valid", data_format=None)
        self.maxpool2 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding="valid", data_format=None)
        self.flatten = Flatten()
        self.dense1 = Dense(100, activation="relu")
        self.logits = Dense(10, activation="linear")
 
        #stateful defense
        self.sd = False
        self.buf_limit = 1000
        self.attacks = 0
        self.buf = None
        self.min_distance = np.inf
        self.examples_processed = 0
        self.distances = []
        self.k = 50
        self.thresh = 3.7

    def set_dist_metric(self, norm=2):
        if norm == 0:
            self.dist_metric = self.linf_norm_dist
        elif norm == 2:
            self.dist_metric = self.l2_norm_dist

    def linf_norm_dist(self, a,b):
        t = a - b
        return np.maximum(np.max(t, axis=(1,2)), np.abs(np.min(t, axis=(1,2))))
        
    def l2_norm_dist(self, a, b):
        t = a-b
        return np.sqrt(np.einsum('kij,kij->k', t, t))

    def attacked(self):
        if self.examples_processed == 0:
            return np.nan

        if self.attacks/self.examples_processed >= 0.001:
            return True
        else:
            return False

    def reset_stateful(self):
        self.attacks = 0
        self.buf = None
        self.min_distance = np.inf
        self.examples_processed = 0
        self.distances = []


    def call(self, x, training=False):
        """
        Call function to evaluate the model.

        :param x: Input to the model
        :return: Prediction of the model
        """
        #if x is not None and x.shape[0] is not None:
        #    self.examples_processed += x.shape[0]

        #print('x.shape', x.shape, ' attacks:', self.attacks)
        if self.sd:
            self.examples_processed += x.shape[0]
            #print('self.examples_processed', self.examples_processed)
            if self.buf is not None and self.buf.shape[0] >= self.k:
                #buf1 = self.buf[-1*self.buf_limit:]
                buf = np.concatenate((np.squeeze(self.buf[-1*self.buf_limit:], axis=-1), np.squeeze(x, axis=-1)))
                x_len = x.shape[0]
                buf_len = self.buf.shape[0]
                if buf_len > self.buf_limit:
                    buf_len = self.buf_limit

                for i in range(x.shape[0]):
                    #d1 = self.dist_metric(np.squeeze(buf[i:], axis=-1), np.squeeze(x[i], axis=-1))
                    #d2 = self.dist_metric(np.squeeze(x[:i], axis=-1), np.squeeze(x[i], axis=-1))
                    #distances = np.sort(np.concatenate((d1, d2)))

                    #comp_buf = np.concatenate((np.squeeze(buf[i:], axis=-1), np.squeeze(x[:i], axis=-1)))
                    endi = -(x_len-i)
                    comp_buf = buf[ endi-self.buf_limit : endi]

                    distances = self.dist_metric(comp_buf, np.squeeze(x[i], axis=-1))

                    #distances.sort()
                    #nzi = np.nonzero(distances)[0][0]
                    #ad = np.average(distances[nzi: nzi+self.k])
                    ##ad = np.average(distances[:self.k])

                    num_zeros = distances.shape[0] - np.count_nonzero(distances)
                    if num_zeros:
                        distances.partition(self.k + num_zeros)
                        temp = distances[:self.k + num_zeros]
                        ad = np.average(temp[temp!=0])
                    else:
                        distances.partition(self.k)
                        ad = np.average(distances[:self.k])

                    if ad <= self.thresh:
                        self.attacks += 1

                    min_temp = ad
                    if min_temp <= self.min_distance:
                        self.min_distance = min_temp

                    self.distances.append(min_temp)

                    #buf = np.concatenate((buf, x[i][np.newaxis]))

            if self.buf is None: 
                self.buf = x
            else:
                self.buf = np.concatenate((self.buf, x))
         
            if self.buf.shape[0] > self.buf_limit*2:
                temp = self.buf[-1*self.buf_limit:]
                gc.enable()
                del self.buf
                gc.collect()
                self.buf = temp

        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = tf.nn.softmax(self.logits(x))
        return x


    def get_config(self):
        return {}

    def from_config(config):
        return TensorFlowModel(**config)

    # def from_config(cls, config):
    #     return cls(**config)

    #def set_weights(self, weights):
    #    self.get_layer('conv2d').set_weights(weights[0])    
    
    #def get_weights(self):
    #    weights = []
    #    weights.append(self.get_layer('conv2d').get_weights)
    #    return weights

def make_callbacks(model_name, save=False):
    """Make list of callbacks for training"""
    callbacks = [EarlyStopping(monitor='val_loss', patience=1)]

    if save:
        callbacks.append(
            ModelCheckpoint(
                f'{model_name}_best_val_loss',
                save_best_only=True,
                save_weights_only=False,
                save_format="tf"))

    return callbacks

#https://github.com/tensorflow/tensorflow/issues/34697
def unpack(model, training_config, weights):
    restored_model = deserialize(model, custom_objects={'TensorFlowModel': TensorFlowModel})
    if training_config is not None:
        restored_model.compile(
            **saving_utils.compile_args_from_training_config(
                training_config
            )
        )
    restored_model.build((None, 28,28,1))
    #print(restored_model.summary())
    restored_model.set_weights(weights)
    return restored_model

# Hotfix function
def make_keras_picklable():
    def __reduce__(self):
        model_metadata = saving_utils.model_metadata(self)
        training_config = model_metadata.get("training_config", None)
        model = serialize(self)
        weights = self.get_weights()
        #print('self.variables', self.variables)
        return (unpack, (model, training_config, weights))

    cls = Model
    cls.__reduce__ = __reduce__

def linf_norm_dist(a,b):
    t = a - b
    return np.abs(t).max(axis=(1,2)) 

def l2_norm_dist(a, b):
    t = a-b
    return np.sqrt(np.einsum('kij,kij->k', t, t))

def choose_best_samples(x_data,y_data, num, k=50, norm=2):
    #features, labels = shuffle(x_data, y_data)
    #return features[:num], labels[:num]

    average_distances = []
    smallest_ad = float('inf')

    if norm == 0:
        dist_metric = linf_norm_dist
        print('set norm 0')
    elif norm == 2:
        dist_metric = l2_norm_dist 

    for i, x in enumerate(x_data):
        #distances = np.sort(np.linalg.norm(np.squeeze(x_data, axis=-1) - np.squeeze(x, axis=-1), axis=(-1,-2), ord=np.inf))
        distances = np.sort(dist_metric(np.squeeze(x_data, axis=-1), np.squeeze(x, axis=-1)))
        nzi = np.nonzero(distances)[0][0]
        ad = np.average(distances[nzi: nzi+k])
        average_distances.append((ad, i))
        if ad < smallest_ad:
            smallest_ad = ad

    average_distances.sort()
    average_distances=average_distances[:x_data.shape[0]-num]

    indicies = [i for _,i in average_distances]
    #indicies.sort(reverse=True)

    x_data=np.delete(x_data, indicies, axis=0)
    y_data=np.delete(y_data, indicies, axis=0)

    print(x_data.shape)
    print(y_data.shape)
    print(smallest_ad)
    #return x_data
    return x_data, y_data


train_model = False

optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

model = TensorFlowModel()
loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

model.compile(optimizer, loss_object)

(x_train, y_train), (x_test, y_test), min_pixel_value, max_pixel_value = load_mnist()

#print(x_train.shape)

model_name = 'test_model'
model_filename = 'test_model.pkl'
model_filename = 'tests/amortized/test_model.pkl'
callbacks = make_callbacks(model_name, False)
#model.save('test_model.h5', save_format='tf')


if train_model:
    make_keras_picklable()
    model.fit(x_train, y_train, batch_size=64, epochs=64, callbacks=callbacks, validation_split=0.1)
    #model.save(f'{model_name}_best_val_loss')
    with open(model_filename, 'wb') as model_file:
        pickle.dump(model, model_file, protocol=pickle.HIGHEST_PROTOCOL)
        print(f'Saved model to {model_filename}')
    
else:
    #model = load_model(f'{model_name}_best_val_loss')
    #model = tf.keras.models.load_model(f'{model_name}_best_val_loss', custom_objects={'TensorFlowModel': TensorFlowModel})
    with open(model_filename, 'rb') as model_file:
        model = pickle.load(model_file)


#print(model.summary())

classifier = TensorFlowV2Classifier(
    model=model,
    loss_object=loss_object,
    #train_step=train_step,
    nb_classes=10,
    input_shape=(28, 28, 1),
    clip_values=(0, 1),
)

if len(sys.argv) >= 9:
    length = int(sys.argv[1])
    reps_arg = int(sys.argv[2])
    chose = int(sys.argv[3])
    sigma = float(sys.argv[4])
    buf_limit = int(sys.argv[5])
    perform_attack = bool(int(sys.argv[6]))
    norm = int(sys.argv[7])
    attack_type = sys.argv[8]

if attack_type == 'amortized':
    amortized_attack = True
elif attack_type == 'vanilla':
    amortized_attack = False

print('length:', length, ' reps:', reps_arg, ' chose:', chose, ' sigma:', sigma, ' buf limit:', buf_limit, ' perform_attack: ', perform_attack, ' norm:', norm, ' amortized attack:', amortized_attack)

model.set_dist_metric(norm)

x_test = x_test[:length]
y_test = y_test[:length]
reps = reps_arg

#x_test = x_test[:1000]
#y_test = y_test[:1000]
#reps = 200
model.buf_limit = buf_limit

if buf_limit == 1000:
    if norm == 0:
        if len(x_test) == 10000:
            model.thresh = 0.928
        elif len(x_test) == 1000:
            model.thresh = 0.903

    elif norm == 1:
        if len(x_test) == 1000:
            model.thresh = 21.0

    elif norm == 2:
        if len(x_test) == 10000:
            model.thresh = 3.45
        elif len(x_test) == 1000:
            model.thresh = 3.25

    else:
        if len(x_test) == 10000:
            model.thresh = 3.7

        if len(x_test) == 1000:
            model.thresh = 5.6

        if len(x_test) == 750:
            model.thresh = 5.7

        if len(x_test) == 500:
            model.thresh = 5.58

        if len(x_test) == 250:
            model.thresh = 5.07

        if len(x_test) == 100:
            model.thresh = 5.27

        if len(x_test) == 50:
            model.thresh = 6.37

elif buf_limit == 10000:
    if norm == 0:
        model.thresh = 0.777
    if norm == 2:
        model.thresh = 2.50

elif buf_limit == 10000:
    if norm == 0:
        model.thresh = 0.719
    elif norm == 2:
        model.thresh = 2.21
    else:
        model.thresh = 2.45

#model.thresh = 5.6

y_col_sum = np.sum(y_test,axis=0)
print('y values before choosing', y_col_sum/np.sum(y_col_sum))

x_test, y_test = choose_best_samples(x_test,y_test, chose, 50, norm=norm)

y_col_sum = np.sum(y_test,axis=0)
print('y values after choosing',  y_col_sum/np.sum(y_col_sum))

#exit()

#x_test, y_test = choose_best_samples(x_test,y_test, 500, 50)

#print(x_test[12].tolist())#

print('len(x_test)', len(x_test))
print('original sum', np.sum(x_test[0]))

if not perform_attack:
    model.sd = True

    start = time.time()
    predictions = classifier.predict(x_test)

    for x in range(reps-1):
        predictions = classifier.predict(x_test)

    accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
    print("Accuracy on benign test examples: {}%".format(accuracy * 100))
    print('model.thresh', model.thresh)
    print('before attack, attack detected: ', model.attacked())
    print('before attack, min distance: ', model.min_distance)
    print('before attack, num attacks: ', model.attacks)
    print('examples_processed', model.examples_processed)
    print('attack percentage', model.attacks/model.examples_processed)


    a = model.distances
    print(0, np.percentile(a, 0))
    print(0.1, np.percentile(a, 0.1))
    print(0.2, np.percentile(a, 0.2))
    #print(0.5, np.percentile(a, 0.5))
    #print(1, np.percentile(a, 1))
    #print(5, np.percentile(a, 5))
    #print(10, np.percentile(a, 10))
    #print(25, np.percentile(a, 25))
    #print(50, np.percentile(a, 50))
    #print(75, np.percentile(a, 75))
    #print(90, np.percentile(a, 90))
    #print(95, np.percentile(a, 95))
    #print(98, np.percentile(a, 98))
    #print(99, np.percentile(a, 99))
    #print(100, np.percentile(a, 100))
    print('length of buf', model.buf.shape[0])
    end = time.time()
    print('time taken:', end - start)
    print()
    exit()


model.sd = False
model.reset_stateful()


if len(sys.argv) >= 3:
    es = float(sys.argv[1])
    tn = int(sys.argv[2])
else:
    es = 0.011
    tn = 7

# Step 6: Generate adversarial test examples
#attack = FastGradientMethod(estimator=classifier, eps=0.35, eps_step=0.08, minimal=True)
#attack = FastGradientMethod(estimator=classifier, eps=0.3, eps_step=es, minimal=True)

#attack = BoundaryAttack(estimator=classifier, targeted=False, delta=0.3, epsilon=0.1, step_adapt=0.66, min_epsilon=0.1, verbose=False)
#attack = BoundaryAttack(estimator=classifier,targeted=False,  verbose=True)

#attack = SimBA(classifier=classifier,targeted=False, attack="dct")

'''
attack = ZooAttack(
    classifier=classifier,
    confidence=0.0,
    targeted=False,
    learning_rate=1e-1,
    max_iter=400,
    binary_search_steps=10,
    initial_const=1e-3,
    abort_early=True,
    use_resize=False,
    use_importance=False,
    nb_parallel=5,
    batch_size=1,
    variable_h=0.01,
)
'''
qe_classifier = QueryEfficientGradientEstimationClassifier(classifier, reps, 1 / sigma, round_samples=1 / 255.0)
qe_classifier.amortized_attack = amortized_attack

#attack = FastGradientMethod(qe_classifier, eps=0.05, eps_step=0.025, batch_size=buf_limit, minimal=True)
attack = ProjectedGradientDescent(qe_classifier, eps=0.05, eps_step=0.025, max_iter=5, batch_size=buf_limit, verbose=False)
print('eps: ', attack.eps, ' eps_step: ', attack.eps_step, ' max_iter', attack.max_iter)
start = time.time()
model.sd = True
x_test_adv = attack.generate(x=x_test)
#x_test_adv = attack.generate(x=x_test, y=y_test)
end = time.time()

print('attack generated', not (x_test==x_test_adv).all())

# Step 7: Evaluate the ART classifier on adversarial test examples

predictions = classifier.predict(x_test_adv)
accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
#print('eps: ', attack.eps, ' eps_step', attack.eps_step)
#print('triangulate: ', attack.triangulate, ' t_n: ', attack.t_n)

for i, v in enumerate(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)):
    if not v:
        print(x_test_adv[i].tolist())
        break

print('length:', length, ' reps:', reps_arg, ' chose:', chose, ' sigma:', sigma, ' buf limit:', buf_limit, ' perform_attack: ', perform_attack, ' norm:', norm, ' amortized attack:', amortized_attack)
print('eps: ', attack.eps, ' eps_step: ', attack.eps_step, ' max_iter', attack.max_iter)


print('time taken:', end - start)
print("Accuracy on adversarial test examples: {}%".format(accuracy * 100))
print('model thresh', model.thresh)
print('after attack, attack detected: ', model.attacked())
print('after attack, min distance: ', model.min_distance)
print('after attack, num attacks: ', model.attacks)
if model.buf is not None:
    print(model.buf.shape)

print('examples_processed', model.examples_processed)

if model.examples_processed != 0:
    print('attack percentage', model.attacks/model.examples_processed)

#asum = np.average(np.sum(np.squeeze(x_test - x_test_adv, axis=-1), axis=(-1,-2)))
#print('average vector sum of differences', asum)

#ad = np.average(np.linalg.norm(np.squeeze(x_test - x_test_adv, axis=-1), axis=(-1,-2), ord=np.inf))
#print('average difference of adversarial samples', ad)

#print('average difference over average original', ad/np.average(np.linalg.norm(np.squeeze(x_test, axis=-1),axis=(-1,-2), ord=np.inf)))

#print('average of the norm of differences over norm of original', np.average(np.linalg.norm(np.squeeze(x_test - x_test_adv, axis=-1), axis=(-1,-2), ord=np.inf)/np.linalg.norm(np.squeeze(x_test, axis=-1), axis=(-1,-2), ord=np.inf)))
#print('average iterations per batch', attack.i/attack.batches)


t = np.squeeze(x_test - x_test_adv, axis=-1)
dist = np.sqrt(np.einsum('kij,kij->k', t, t))
o = np.squeeze(x_test, axis=-1)
norm = np.sqrt(np.einsum('kij,kij->k', o, o))
print('average l2 norm of difference over l2 norm of original', np.average(dist/norm))


dist = linf_norm_dist(np.squeeze(x_test, axis=-1),  np.squeeze( x_test_adv, axis=-1))
o = np.squeeze(x_test, axis=-1)
norm = np.abs(o).max(axis=(1,2))
print('average l-infinity norm of difference over l-infinity norm of original', np.average(dist/norm))

#print('average of the norm of the (norm of differences over original)', np.average(np.linalg.norm(np.linalg.norm(np.squeeze(x_test - x_test_adv, axis=-1), axis=(-1,-2), ord=np.inf)[:, None, None]/np.squeeze(x_test, axis=-1), axis=(-1,-2) )))

#print(x_test[12].tolist())
#print(x_test_adv[0].tolist())#also 13
print()
