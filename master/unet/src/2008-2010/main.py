import os
# import model
# import myModel
from model import unet
# import data
from data import trainGenerator, validGenerator, testGenerator, saveResult
from keras.callbacks import ModelCheckpoint
from keras.utils import plot_model
import matplotlib.pyplot as plt


import time
import datetime

now = datetime.datetime.now()
curr_time = now.strftime('%Y%m%d_%H%M%S')

start = time.perf_counter()

# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
# os.environ["CUDA_VISIBLE_DEVICES"] = "2,3,4,5,6,7"
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,4,5,6"


data_gen_args = dict(rotation_range=0.5,
                    width_shift_range=0.1,
                    height_shift_range=0.1,
                    shear_range=0.1,
                    zoom_range=0.1,
                    horizontal_flip=True,
                    vertical_flip=True,
                    fill_mode='constant',
                    cval=0.0)
# fill_mode : https://book-read-yoshi.hatenablog.com/entry/2021/09/17/data_augmentation/imagedatagenerator

trainGene = trainGenerator(2,'data/membrane/train','image','label',data_gen_args,save_to_dir = None)
validGene = validGenerator(2,'data/membrane/valid','image','label',data_gen_args,save_to_dir = None)

unet_model = unet()
model_checkpoint = ModelCheckpoint('data/membrane/models/unet_'+ curr_time +'.hdf5', monitor='val_loss',verbose=1, save_best_only=True)
"""
no validation steps
# history = unet_model.fit(trainGene, steps_per_epoch=1, epochs=2, callbacks=[model_checkpoint])
"""

# unet_model.fit_generator(trainGene,steps_per_epoch=300,epochs=1,callbacks=[model_checkpoint])
# history = unet_model.fit(trainGene, steps_per_epoch=1, epochs=2, validation_steps=1, validation_split=0.1, callbacks=[model_checkpoint])

"""
with validation steps
"""
# VALIDATION_STEPS = 20
# STEPS_PER_EPOCH = 10
# EPOCHS = 200
# VERBOSE = 2

history = unet_model.fit(trainGene, validation_data=validGene, \
                            validation_steps=50, steps_per_epoch=1000, epochs=20, verbose=2, \
                            shuffle=True, callbacks=[model_checkpoint],)

# history = unet_model.fit_generator(trainGene, validation_data=validGene, validation_steps=3, steps_per_epoch=step_epoch, epochs=epochs, verbose=2, shuffle=True, callbacks=[model_checkpoint,tensorboard,history])
history_dict = history.history
print('history_dict.keys():', history_dict.keys())
# history_dict.keys(): dict_keys(['loss', 'accuracy', 'val_loss', 'val_accuracy'])

plot_model(unet_model, to_file="/data/Users/izumi/unet/2008-2010/data/membrane/logs/"+curr_time+"_model.png")




os.makedirs("data/membrane/results/" + curr_time, exist_ok=True)

"""
plot accuracy
"""
plt.figure()
plt.plot(history.history["accuracy"], label="acc")
plt.plot(history.history["val_accuracy"], label="val_acc")
plt.ylabel("accuracy")
plt.xlabel("epoch")
plt.legend(loc="best")
# plt.show()
plt.savefig("data/membrane/results/" + curr_time + "/acc.png")

"""
plot loss
"""
plt.figure()
plt.plot(history.history["loss"], label="loss")
plt.plot(history.history["val_loss"], label="val_loss")
plt.ylabel("loss")
plt.xlabel("epoch")
plt.legend(loc="best")
# plt.show()
plt.savefig("data/membrane/results/"+ curr_time  + "/loss.png")



testGene = testGenerator("data/membrane/test")
# results = unet_model.predict_generator(testGene,24,verbose=1)

results = unet_model.predict(testGene,24,verbose=1)
saveResult("data/membrane/results/" + curr_time, results)

print("Running time :", time.perf_counter() - start)

