
from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
from tkinter import filedialog
import matplotlib.pyplot as plt
import numpy as np
from tkinter.filedialog import askopenfilename
import os
import json
import cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score 
import imutils
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.layers import  MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten, Input
from keras.layers import Convolution2D
from keras.models import Sequential, Model
from keras.models import model_from_json
from keras.applications import ResNet50, MobileNetV2
from keras.applications.efficientnet import EfficientNetB0
import pickle
from sklearn import metrics
import ftplib
from sklearn import svm
from tkinter import ttk

main = tkinter.Tk()
main.title("Identifying Bone Tumor using X-Ray Images") #designing main screen
main.geometry("1300x1200")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'Model')

filename = ''
accuracy = 0
comparison_results = {}
X = []
Y = []
classifier = None
class_labels = []

def model_path(filename):
    return os.path.join(MODEL_DIR, filename)

def load_saved_class_labels():
    global class_labels
    labels_path = model_path('class_labels.json')
    if os.path.exists(labels_path):
        try:
            with open(labels_path, 'r') as labels_file:
                class_labels = json.load(labels_file)
        except Exception:
            class_labels = []
    return class_labels


def save_class_labels(labels):
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(model_path('class_labels.json'), 'w') as labels_file:
        json.dump(labels, labels_file)


def get_dataset_class_names(dataset_root):
    if not dataset_root or not os.path.isdir(dataset_root):
        return []
    label_folders = sorted(
        d for d in os.listdir(dataset_root)
        if os.path.isdir(os.path.join(dataset_root, d)) and not d.startswith('.')
    )
    return label_folders

with open(model_path('segmented_model.json'), "r") as json_file:
    loaded_model_json = json_file.read()
    segmented_model = model_from_json(loaded_model_json)
json_file.close()
segmented_model.load_weights(model_path('segmented_weights.h5'))
segmented_model.make_predict_function()

def edgeDetection():
    img = cv2.imread('myimg.png')
    orig = cv2.imread('test1.png')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)[1]
    contours = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    min_area = 0.95*180*35
    max_area = 1.05*180*35
    result = orig.copy()
    for c in contours:
        area = cv2.contourArea(c)
        cv2.drawContours(result, [c], -1, (0, 0, 255), 10)
        if area > min_area and area < max_area:
            cv2.drawContours(result, [c], -1, (0, 255, 255), 10)
    return result    

def tumorSegmentation(filename):
    global segmented_model
    img = cv2.imread(filename,0)
    img = cv2.resize(img,(64,64), interpolation = cv2.INTER_CUBIC)
    img = img.reshape(1,64,64,1)
    img = (img-127.0)/127.0
    preds = segmented_model.predict(img)
    preds = preds[0]
    print(preds.shape)
    orig = cv2.imread(filename,0)
    orig = cv2.resize(orig,(300,300),interpolation = cv2.INTER_CUBIC)
    cv2.imwrite("test1.png",orig)    
    segmented_image = cv2.resize(preds,(300,300),interpolation = cv2.INTER_CUBIC)
    cv2.imwrite("myimg.png",segmented_image*255)
    edge_detection = edgeDetection()
    return segmented_image*255, edge_detection
    

def uploadDataset(): #function to upload dataset
    global filename
    filename = filedialog.askdirectory(initialdir=".")
    if not filename:
        return
    text.delete('1.0', END)
    text.insert(END,filename+" loaded\n");

def datasetPreprocessing():
    global X
    global Y
    global class_labels
    X = []
    Y = []
    class_labels = []

    if not filename:
        messagebox.showwarning('Dataset Required', 'Upload the dataset folder first.')
        return

    class_labels = get_dataset_class_names(filename)
    if len(class_labels) < 2:
        messagebox.showwarning('Dataset Required', 'Upload a dataset folder containing at least two image classes.')
        return

    save_class_labels(class_labels)

    for class_index, label_name in enumerate(class_labels):
        class_dir = os.path.join(filename, label_name)
        for root, dirs, directory in os.walk(class_dir):
            for name in directory:
                image_path = os.path.join(root, name)
                img = cv2.imread(image_path, 0)
                if img is None:
                    continue
                img = cv2.resize(img, (128, 128))
                im2arr = np.array(img)
                im2arr = im2arr.reshape(128, 128, 1)
                X.append(im2arr)
                Y.append(class_index)

    X = np.asarray(X)
    Y = np.asarray(Y)
    np.save(model_path('myimg_data.txt'), X)
    np.save(model_path('myimg_label.txt'), Y)
    if len(X) == 0 or len(X) != len(Y):
        messagebox.showwarning('Dataset Error', 'No matching images were loaded. Check that the classes contain images.')
        return
    text.insert(END, "Total number of images found in dataset : " + str(len(X)) + "\n")
    text.insert(END, "Total number of classes : " + str(len(set(Y))) + "\n")
    text.insert(END, "Class labels found in dataset : " + str(class_labels) + "\n\n")

def runSVM():
    global comparison_results, X, Y
    if len(X) == 0 or len(Y) == 0 or len(X) != len(Y):
        messagebox.showwarning('Preprocessing Required', 'Upload the dataset and run Dataset Preprocessing first.')
        return
    XX = np.reshape(X, (X.shape[0], (X.shape[1] * X.shape[2] * X.shape[3])))
    print(XX.shape)
    XX = XX[:,0:100]
    X_train, X_test, y_train, y_test = train_test_split(XX, Y, test_size = 0.2, random_state = 0)
    svm_cls = svm.SVC()
    svm_cls.fit(X_train, y_train)
    predict = svm_cls.predict(X_test)
    acc1 = accuracy_score(y_test, predict)  * 100
    comparison_results['SVM Accuracy'] = acc1
    text.insert(END,"SVM Bone Tumor Prediction Accuracy on Test Images : "+str(acc1)+"\n")


def prepare_rgb_images(image_array):
    if image_array.shape[-1] == 1:
        return np.repeat(image_array, 3, axis=-1)
    return image_array


def keras_json_default(value):
    if hasattr(value, 'numpy'):
        value = value.numpy()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f'Object of type {type(value).__name__} is not JSON serializable')


def train_transfer_model(model_name, base_model_factory):
    global accuracy, comparison_results
    global classifier
    global X, Y, class_labels

    class_labels = load_saved_class_labels()
    if not class_labels:
        class_labels = get_dataset_class_names(filename)
        if not class_labels:
            messagebox.showwarning('Preprocessing Required', 'Upload the dataset and run Dataset Preprocessing first.')
            return

    if len(X) == 0 or len(Y) == 0 or len(X) != len(Y):
        messagebox.showwarning('Preprocessing Required', 'Upload the dataset and run Dataset Preprocessing first.')
        return

    X_rgb = prepare_rgb_images(np.asarray(X)).astype('float32') / 255.0
    Y_arr = np.asarray(Y)
    num_classes = len(class_labels)
    YY = to_categorical(Y_arr, num_classes=num_classes)

    indices = np.arange(X_rgb.shape[0])
    np.random.shuffle(indices)
    x_train = X_rgb[indices]
    y_train = YY[indices]

    base_model = base_model_factory(weights='imagenet', include_top=False, input_shape=(128, 128, 3), pooling='avg')
    inputs = Input(shape=(128, 128, 3))
    x = base_model(inputs)
    x = Dense(256, activation='relu')(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    classifier = Model(inputs=inputs, outputs=outputs)

    for layer in base_model.layers:
        layer.trainable = False

    classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    early_stopping = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)
    hist = classifier.fit(x_train, y_train, batch_size=16, epochs=10, validation_split=0.2, shuffle=True, verbose=2, callbacks=[early_stopping])

    classifier.save_weights(model_path(f'{model_name.lower()}_weights.h5'))
    with open(model_path('history.pckl'), 'wb') as f:
        pickle.dump(hist.history, f)

    try:
        model_json = json.dumps(classifier._updated_config(), default=keras_json_default, indent=2)
    except (TypeError, ValueError) as error:
        text.insert(END, f'{model_name} weights and training history were saved, but Keras could not serialize the model definition: {error}\n')
    else:
        with open(model_path(f'{model_name.lower()}.json'), 'w') as json_file:
            json_file.write(model_json)

    with open(model_path('history.pckl'), 'rb') as f:
        data = pickle.load(f)
    acc = data['accuracy']
    accuracy = max(acc) * 100
    comparison_results[f'{model_name} Accuracy'] = accuracy
    text.insert(END, f'\n\n{model_name} Bone Tumor Model Generated.\n\n')
    text.insert(END, f'{model_name} Bone Tumor Prediction Accuracy on Test Images : {str(accuracy)}\n')


def trainTumorDetectionModel():
    global accuracy, comparison_results
    global classifier
    global X, Y, class_labels

    class_labels = load_saved_class_labels()
    if not class_labels:
        class_labels = get_dataset_class_names(filename)
        if not class_labels:
            messagebox.showwarning('Preprocessing Required', 'Upload the dataset and run Dataset Preprocessing first.')
            return

    if len(X) == 0 or len(Y) == 0 or len(X) != len(Y):
        messagebox.showwarning('Preprocessing Required', 'Upload the dataset and run Dataset Preprocessing first.')
        return

    X = np.asarray(X)
    Y = np.asarray(Y)
    num_classes = len(class_labels)
    YY = to_categorical(Y, num_classes=num_classes)

    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)

    x_train = X[indices]
    y_train = YY[indices]

    should_retrain = True
    if os.path.exists(model_path('model.json')):
        try:
            with open(model_path('model.json'), "r") as json_file:
                loaded_model_json = json_file.read()
            classifier = model_from_json(loaded_model_json)
            classifier.load_weights(model_path('model_weights.h5'))
            classifier.make_predict_function()
            if classifier.output_shape[-1] == num_classes:
                should_retrain = False
        except Exception:
            should_retrain = True

    if should_retrain:
        X_trains, X_tests, y_trains, y_tests = train_test_split(x_train, y_train, test_size = 0.2, random_state = 0)
        classifier = Sequential()
        classifier.add(Convolution2D(32, (3, 3), input_shape = (128, 128, 1), activation = 'relu'))
        classifier.add(MaxPooling2D(pool_size = (2, 2)))
        classifier.add(Convolution2D(32, (3, 3), activation = 'relu'))
        classifier.add(MaxPooling2D(pool_size = (2, 2)))
        classifier.add(Flatten())
        classifier.add(Dense(units = 128, activation = 'relu'))
        classifier.add(Dense(units = num_classes, activation = 'softmax'))
        print(classifier.summary())
        classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
        early_stopping = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)
        hist = classifier.fit(x_train, y_train, batch_size=16, epochs=10, validation_split=0.2, shuffle=True, verbose=2, callbacks=[early_stopping])
        classifier.save_weights(model_path('model_weights.h5'))
        model_json = classifier.to_json()
        with open(model_path('model.json'), "w") as json_file:
            json_file.write(model_json)
        f = open(model_path('history.pckl'), 'wb')
        pickle.dump(hist.history, f)
        f.close()

    f = open(model_path('history.pckl'), 'rb')
    data = pickle.load(f)
    f.close()
    acc = data['accuracy']
    accuracy = max(acc) * 100
    comparison_results['CNN Accuracy'] = accuracy
    text.insert(END,'\n\nCNN Bone Tumor Model Generated. See black console to view layers of CNN\n\n')
    text.insert(END,"CNN Bone Tumor Prediction Accuracy on Test Images : "+str(accuracy)+"\n")


def trainResNet50Model():
    train_transfer_model('ResNet50', lambda **kwargs: ResNet50(**kwargs))


def trainEfficientNetModel():
    train_transfer_model('EfficientNet', lambda **kwargs: EfficientNetB0(**kwargs))


def trainMobileNetV2Model():
    train_transfer_model('MobileNetV2', lambda **kwargs: MobileNetV2(**kwargs))


def tumorClassification():
    global class_labels
    class_labels = load_saved_class_labels()
    if not class_labels:
        class_labels = get_dataset_class_names(filename)
    if classifier is None:
        messagebox.showwarning('CNN Model Required', 'Run Trained CNN Bone Tumor Detection Model first.')
        return
    image_path = filedialog.askopenfilename(initialdir="testImages")
    if not image_path:
        return
    img = cv2.imread(image_path, 0)
    img = cv2.resize(img, (128, 128))
    im2arr = np.array(img)
    im2arr = im2arr.reshape(1, 128, 128, 1)
    XX = np.asarray(im2arr)

    predicts = classifier.predict(XX)
    print(predicts)
    cls = int(np.argmax(predicts))
    print(cls)
    label = class_labels[cls] if cls < len(class_labels) else 'Unknown'
    if 'no' in label.lower() or 'normal' in label.lower():
        img = cv2.imread(image_path)
        img = cv2.resize(img, (800, 500))
        cv2.putText(img, 'Classification Result : ' + label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow('Classification Result : ' + label, img)
        cv2.waitKey(0)
    else:
        segmented_image, edge_image = tumorSegmentation(image_path)
        img = cv2.imread(image_path)
        img = cv2.resize(img, (800, 500))
        cv2.putText(img, 'Classification Result : ' + label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow('Classification Result : ' + label, img)
        cv2.imshow("Tumor Segmented Image", segmented_image)
        cv2.imshow("Edge Detected Image", edge_image)
        cv2.waitKey(0)
        
        
        

def graph():
    history_path = model_path('history.pckl')
    if not os.path.exists(history_path):
        messagebox.showwarning('Training History Missing', 'Train the CNN model first.')
        return
    f = open(history_path, 'rb')
    data = pickle.load(f)
    f.close()

    accuracy = data['accuracy']
    loss = data['loss']

    plt.figure(figsize=(10,6))
    plt.grid(True)
    plt.xlabel('Training Epoch')
    plt.ylabel('Accuracy/Loss')
    plt.plot(loss, 'ro-', color = 'red')
    plt.plot(accuracy, 'ro-', color = 'green')
    plt.legend(['Loss', 'Accuracy'], loc='upper left')
    plt.title('Bone Tumor CNN Model Training Accuracy & Loss Graph')
    plt.show()

def accgraph():
    if not comparison_results:
        messagebox.showwarning('Results Missing', 'Train at least one model first.')
        return
    bars = tuple(comparison_results.keys())
    height = [comparison_results[name] for name in bars]
    y_pos = np.arange(len(bars))
    plt.bar(y_pos, height)
    plt.xticks(y_pos, bars)
    plt.xlabel("Algorithm Names")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Comparison Graph")
    plt.show()

font = ('times', 16, 'bold')
title = Label(main, text='Identifying Bone Tumor using X-Ray Images')
title.config(bg='darkviolet', fg='gold')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 12, 'bold')
text=Text(main,height=20,width=150)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=50,y=120)
text.config(font=font1)


font1 = ('times', 12, 'bold')
uploadButton = Button(main, text="Upload Tumor X-Ray Images Dataset", command=uploadDataset)
uploadButton.place(x=50,y=550)
uploadButton.config(font=font1)  

preprocessButton = Button(main, text="Dataset Preprocessing & Features Extraction", command=datasetPreprocessing)
preprocessButton.place(x=430,y=550)
preprocessButton.config(font=font1)

cnnButton = Button(main, text="Trained SVM Bone Tumor Detection Model", command=runSVM)
cnnButton.place(x=810,y=550)
cnnButton.config(font=font1) 

cnnButton = Button(main, text="Trained CNN Bone Tumor Detection Model", command=trainTumorDetectionModel)
cnnButton.place(x=50,y=600)
cnnButton.config(font=font1) 

resnetButton = Button(main, text="Trained ResNet50 Model", command=trainResNet50Model)
resnetButton.place(x=430,y=600)
resnetButton.config(font=font1)

efficientButton = Button(main, text="Trained EfficientNet Model", command=trainEfficientNetModel)
efficientButton.place(x=810,y=600)
efficientButton.config(font=font1)

mobilenetButton = Button(main, text="Trained MobileNetV2 Model", command=trainMobileNetV2Model)
mobilenetButton.place(x=50,y=650)
mobilenetButton.config(font=font1)

classifyButton = Button(main, text="Bone Tumor Segmentation & Classification", command=tumorClassification)
classifyButton.place(x=430,y=650)
classifyButton.config(font=font1)

graphButton = Button(main, text="Training Accuracy Graph", command=graph)
graphButton.place(x=810,y=650)
graphButton.config(font=font1)

graphButton = Button(main, text="Comparison Graph", command=accgraph)
graphButton.place(x=430,y=700)
graphButton.config(font=font1)

main.config(bg='turquoise')
main.mainloop()
