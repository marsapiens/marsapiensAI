from tkinter import * 
from tkinter import filedialog, ttk, messagebox
from PIL import ImageTk, Image
Image.MAX_IMAGE_PIXELS = 1000000000000000000
from tkinter.filedialog import askopenfilename
import os
import numpy as np
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2,40).__str__()
import cv2

root = Tk()
root.title('Marsapiens')
root.iconbitmap('./Marsico.ico')
root.resizable(False, False)
root.configure(bg='#2d2c2c')
# root.wm_attributes('-transparentcolor','gray')  #LOL esto hace transparente TODO
image2 =Image.open('./Marsimg2.PNG')
image2 = image2.resize((350,350), Image.BILINEAR)

image1 = ImageTk.PhotoImage(image2)
w = image1.width()+340
h = image1.height()+250
root.geometry('%dx%d+0+0' % (w,h))





resol = StringVar()    
def getinput():
    global res
    reso = str(resol.get())
    
resol.set("224") # default value

getinput = ttk.OptionMenu(root, resol, '224','112', '224', '448').place(x=50, y=430)

l_message_res = Label(root, text='Detections dimension (px)', fg='white', bg='#2d2c2c',font=(None, 9))
l_message_res.place(x=47, y=405)


var = StringVar()

def sel():
    global var
    selection = str(var.get())
    
var.set('None') # default value

mascaralist= ttk.OptionMenu(root, var, 'None','None', 'High', 'Medium', 'Low').place(x=520, y=430)
print(var)
l_message_sens = Label(root, text='Cleaning sensitivity', fg='white', bg='#2d2c2c',font=(None, 9))
l_message_sens.place(x=515, y=405)


def selectpath():
    global read_path
    
    read_path = filedialog.askopenfilename(title='Open', filetypes=(("JPEG2000 files", "*.JP2"),("PNG files", "*.PNG")))
    filename = os.path.basename(read_path)
    l_filename = Label(root, text=filename, fg='black', font=(None, 9))
    l_filename.place(x=155, y=357)



    
l_image = Label(root, image=image1, borderwidth=0)
l_image.pack()

    
b_path = Button(root,width = 12, text='Open image',relief='raised' , command=selectpath) #relief modes: sunken, raised, flat
b_path.place(x=50, y=355)

ent1=Entry(root,width = 43, font=9 , relief='sunken', state='disabled')
ent1.place(relx=0, x=150,y=354)

sav = '0'
save = IntVar()
def savepng():
    global sav
    sav = int(save.get())
    print(sav)
    
save.set('0')

l_save_png = Label(root, text='Save converted PNG', fg='white', bg='#2d2c2c',font=(None, 9))
l_save_png.place(x=66, y=496)

ttk.Style().configure("TCheckbutton", padding=0, relief="flat",
   background='#2d2c2c')

b_savepng = ttk.Checkbutton(root, variable=save, onvalue='1', offvalue='0', command = savepng).place(x=50,y=497)


    
def analizar():
    
    
    global im_full

    
    name = os.path.basename(read_path)[:-4]
    PNG_dir = './png_marsapiens/'
    try:
        os.mkdir('./temp')
        os.mkdir('./temp/chunks')
    except OSError as e:
        print('')
    try:
        os.mkdir('./png_marsapiens')
    except OSError as e:
        print('')  

    im = cv2.imread(read_path)



    
    if read_path[-3:] == 'JP2':
        cv2.imwrite(PNG_dir + name + '.PNG',im)
        im_full = Image.open(PNG_dir + name + '.PNG')
        infile = PNG_dir + name + '.PNG'
    else:
        im = Image.open(read_path)
        im_full = im
        infile = read_path

    
    chunks_dir = './temp/chunks'




    print('JP2 to PNG succesful')
    
    global resol
    
    reso = resol.get()
    if reso == '112':
        res = 112
    elif reso == '224':
        res = 224
    elif reso == '448':
        res = 448
    h = im_full.size[0]
    w = im_full.size[1]
    nh = int((h/res))*res
    nw = int((w/res))*res
    box = (0,0,nw,nh)
    im_crop = im_full.crop(box)
    print(im_crop.size)
    n_chunks = nw*nh/(res*res)
    n_chunks = int(n_chunks)
    chunks_vert = int(nh/res)
    chunks_hor = int(nw/res)


    start_num = 1


    def crop(infile,height,width):
        im = Image.open(infile)
        imgwidth, imgheight = im.size
        for i in range(imgheight//height):
            for j in range(imgwidth//width):
                box = (j*width, i*height, (j+1)*width, (i+1)*height)
                yield im.crop(box)

    

    for k,piece in enumerate(crop(infile,res,res),start_num):
        img = Image.new('L', (res,res), 255)
        img.paste(piece)
        cname = str(k)
        cname =cname.zfill(10)
        path = os.path.join(chunks_dir, cname + '.JPG')

        img.save(path)


    import tensorflow as tf

    #Bloque prediccion
    pred_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255, samplewise_std_normalization=True)
    pred_generator = pred_datagen.flow_from_directory(
        "./temp/",
        target_size=(224, 224),
        color_mode="grayscale",
        batch_size=1,
        shuffle = False,
        class_mode='input')
    init = tf.keras.initializers.RandomNormal(mean=0.5,stddev=0.5, seed=None)

    model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(64,(3,3),activation="relu", input_shape=(224,224,1),
                           kernel_initializer=init
                          ),
    tf.keras.layers.MaxPooling2D((3,3)),
    tf.keras.layers.Conv2D(16,(3,3),activation="relu"),
    tf.keras.layers.MaxPooling2D((3,3)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid")
    ])


    model.summary()

    model.load_weights("./wts/wts.h5")

    model.compile(loss='binary_crossentropy',
              optimizer='SGD',
              metrics=['acc'])

    pred = model.predict_generator(pred_generator, steps=None, max_queue_size=10, workers=8, use_multiprocessing=False, verbose=1)  
    predicted = np.argmax(pred, axis=1)
    directorio = []
    for i in range(n_chunks): 

        image_batch, classes_batch = next(pred_generator)
        predicted_batch = model.predict(image_batch)
        for k in range(0,image_batch.shape[0]):
            image = image_batch[k]

            pred = predicted_batch[k]
            pred = max(pred)
            if pred >= 0.01:
                pred = 1
            else:
                pred = 0
            directorio.append(pred)
    preds = np.asarray(directorio)
    mask = np.reshape(preds,(chunks_hor,chunks_vert))
    mask = mask.astype('uint8')
    mascara = mask
    valor_mascara = var.get()
    if valor_mascara == 'None':
        mascara = mask
    elif valor_mascara == 'High':
        mascara = [[0 for i in linea] for linea in mask]
        for i in range(len(mask) - 2):
            submatriz = mask[i:i+3]
            for e in range(len(submatriz[0]) - 2):
                suma = sum([sum(linesub[e:e+3]) for linesub in submatriz])
                if suma >= 3:
                    for d in range(3):
                        mascara[i+d][e:e+3] = [1,1,1]
    elif valor_mascara == 'Medium':
        mascara = [[0 for i in linea] for linea in mask]
        for i in range(len(mask) - 2):
            submatriz = mask[i:i+3]
            for e in range(len(submatriz[0]) - 2):
                suma = sum([sum(linesub[e:e+3]) for linesub in submatriz])
                if suma >= 5:
                    for d in range(3):
                        mascara[i+d][e:e+3] = [1,1,1]
    elif valor_mascara == 'Low':        
        mascara = [[0 for i in linea] for linea in mask]
        for i in range(len(mask) - 2):
            submatriz = mask[i:i+3]
            for e in range(len(submatriz[0]) - 2):
                suma = sum([sum(linesub[e:e+3]) for linesub in submatriz])
                if suma >= 7:
                    for d in range(3):
                        mascara[i+d][e:e+3] = [1,1,1]
                        
    mascara = np.asarray(mascara, dtype="uint8")
    mascara = mascara.repeat(res, axis=0).repeat(res, axis=1)

    maskrgb = mascara

    final = cv2.imread(infile, cv2.IMREAD_GRAYSCALE)
    resized = cv2.resize(final, (nh,nw), interpolation = cv2.INTER_AREA)
    resized_arr = np.asarray(resized)

    dst_arr = cv2.addWeighted(resized_arr,0.7,maskrgb,70,0)
    maskpng = cv2.addWeighted(maskrgb,0.7,maskrgb,70,0)
    
    valor_mascara = var.get()
    try:
        os.mkdir('./detections/')
    except OSError as e:
        print('')
        
    cv2.imwrite('./detections/' + name + valor_mascara +'_' + reso +   '_detect.JP2', cv2.cvtColor(dst_arr, cv2.COLOR_RGB2BGR))
    cv2.imwrite('./detections/' + name +'_Mask_' + valor_mascara +'_' + reso + '_detect.PNG', cv2.cvtColor(maskpng, cv2.COLOR_RGB2BGR))
    pathfinal = os.getcwd() +'\detections'
    pathfinal = str(pathfinal)
    
    
    import shutil
    global sav
    global save
    print('sav',sav)
    if sav == 1.:
        shutil.rmtree('./temp')
        path_png = os.getcwd() + '\png_marsapiens'
        messagebox.showinfo('Analysis complete', 'The image detections are saved in ' + pathfinal + '\n' + '\n'+'The image converted to PNG is in ' + path_png)
    else:
        shutil.rmtree('./temp')
        os.remove('./png_marsapiens/'+ name + '.PNG')
        # shutil.rmtree('./png_marsapiens')
        messagebox.showinfo('Analysis complete', 'The image detections are saved in ' + pathfinal)

    l_filename = Label(root, text='                                                                                ', fg='black', font=(None, 9))
    l_filename.place(x=155, y=357)
    
    
    

    
    



b_analyze = Button(root, text='Analyze', height=1, width = 17, relief='raised', font=(None, 18), bg='#D23F04', fg='white', command = analizar)
b_analyze.place(relx=0.5, y=520, anchor=CENTER)
root.mainloop()