import json
from PIL import Image
import tkinter as tk
from tkinter import filedialog
from torchvision import models
import torchvision.transforms as transforms

window = tk.Tk()
window.title('Image Classifier')
window.geometry("300x300")
window.configure(bg='white')

image_class_index = json.load(open('image_class_index.json'))
model = models.densenet121(pretrained=True)
model.eval()

pred = tk.Label(window, text="LABEL")
pred.place(relx=0.45, y=50)


def getImage():
    global window, pred
    pred.place_forget()
    import_file_path = filedialog.askopenfilename()
    class_id, class_name = get_prediction(import_file_path)
    pred = tk.Label(window, text=class_name)
    pred.place(relx=0.45, y=50)


def transform_image(image_bytes):
    my_transforms = transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(
                                            [0.485, 0.456, 0.406],
                                            [0.229, 0.224, 0.225])])
    image = Image.open(image_bytes)
    return my_transforms(image).unsqueeze(0)


def get_prediction(image_bytes):
    tensor = transform_image(image_bytes=image_bytes)
    outputs = model.forward(tensor)
    _, y_hat = outputs.max(1)
    predicted_idx = str(y_hat.item())
    return image_class_index[predicted_idx]


def clearScreen():
    global window
    window.destroy()


button_1 = tk.Button(text='Upload Image', bg='#000000', fg='#ffffff', command=getImage)
button_1.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
button_1 = tk.Button(text='Close Window', bg='#000000', fg='#ffffff', command=clearScreen)
button_1.place(relx=0.5, rely=0.6, anchor=tk.CENTER)

window.mainloop()