from LinearDeformer import *
from BiomechDeformer import *
from CircleMaskFinder import *
from EyePreserve import *
from SyntheticIrisGenerator import *
from tkinter import Frame, Button, Scale, IntVar, Checkbutton, Label, Tk, filedialog, HORIZONTAL, Radiobutton
from PIL import Image, ImageTk
import torch
import sys


wide_size_original = (400, 300)
square_size_original = (300, 300)

wide_size = wide_size_original
square_size = square_size_original

if len(sys.argv) > 1 and sys.argv[1] == 'cuda':
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

ir_ratio = (square_size[0]/2) / ((square_size[0]/2) - (square_size[0]/16))

root = Tk()
circlemaskfinder = CircleMaskFinder(mask_net_path = "./models/nestedsharedatrousresunet-156-0.026496-maskIoU-0.943222.pth", circle_net_path = './models/convnext_tiny-1076-0.030622-maskIoU-0.938355.pth', eyelid_net_path = "./models/nestedsharedatrousresunet-1935-0.036982-maskIoU-0.946174-insideeyelid.pth", device=device)
ldeformer = LinearDeformer(device=device)
bdeformer = BiomechDeformer(device=device)
eyepreserve = EyePreserve(net_path="./models/0007-val_loss-22.957276336365826+-0.0-val_bit_match-81.10588183031244+-0.0-val_linear_bit_match-75.13033321984577+-0.0.pth", device=device)
generator = SyntheticIrisGenerator(device=device)

im = generator.generate_image().convert('L')
ow, oh = im.size
im, w_pad, h_pad = circlemaskfinder.fix_image(im)
mask, back_mask, pxyr, ixyr = circlemaskfinder.segmentAndCircApprox(im.convert("L"))
w, h = im.size
im = im.crop((w_pad, h_pad, w_pad + ow, h_pad + oh))
mask = mask.crop((w_pad, h_pad, w_pad + ow, h_pad + oh))
back_mask = back_mask.crop((w_pad, h_pad, w_pad + ow, h_pad + oh))
pxyr[0] = pxyr[0] - w_pad
ixyr[0] = ixyr[0] - w_pad
pxyr[1] = pxyr[1] - h_pad
ixyr[1] = ixyr[1] - h_pad
alpha = pxyr[2]/ixyr[2]

frame = Frame(root)

modelvar0=Label(frame, text="Generated Image", font=('Arial', 20))
modelvar0.grid(row=0, column=0)

modelvar1=Label(frame, text="Linear Deformation", font=('Arial', 20))
modelvar1.grid(row=0, column=1)

modelvar2=Label(frame, text="Biomechanical Deformation", font=('Arial', 20))
modelvar2.grid(row=0, column=2)

modelvar2=Label(frame, text="EyePreserve", font=('Arial', 20))
modelvar2.grid(row=0, column=3)

tkimage1 = ImageTk.PhotoImage(im.resize(square_size))
myvar1=Label(frame,image = tkimage1)
myvar1.image = tkimage1
myvar1.grid(row=1, column=0)

ld_im = ldeformer.linear_deform(im.copy(), mask.copy(), pxyr.copy(), ixyr.copy(), alpha)
tkimage2 = ImageTk.PhotoImage(ld_im.resize(square_size))
myvar2=Label(frame,image = tkimage2)
myvar2.image = tkimage2
myvar2.grid(row=1, column=1)

bd_im = bdeformer.biomechanical_deform(im.copy(), mask.copy(), pxyr.copy(), ixyr.copy(), alpha)
tkimage3 = ImageTk.PhotoImage(bd_im.resize(square_size))
myvar3=Label(frame,image = tkimage3)
myvar3.image = tkimage3
myvar3.grid(row=1, column=2)

dnet_im = eyepreserve.deform_crop_with_alpha(im.copy(), mask.copy(), back_mask.copy(), pxyr.copy(), ixyr.copy(), alpha, device)
tkimage4 = ImageTk.PhotoImage(dnet_im.resize(square_size))
myvar4=Label(frame,image = tkimage4)
myvar4.image = tkimage4
myvar4.grid(row=1, column=3)

def randomizeimage():
    global im, mask, back_mask, pxyr, ixyr, alpha
    im = generator.generate_image().convert("L")
    ow, oh = im.size
    im, w_pad, h_pad = circlemaskfinder.fix_image(im)
    mask, back_mask, pxyr, ixyr = circlemaskfinder.segmentAndCircApprox(im.convert("L"))
    w, h = im.size
    im = im.crop((w_pad, h_pad, w_pad + ow, h_pad + oh))
    mask = mask.crop((w_pad, h_pad, w_pad + ow, h_pad + oh))
    back_mask = back_mask.crop((w_pad, h_pad, w_pad + ow, h_pad + oh))
    pxyr[0] = pxyr[0] - w_pad
    ixyr[0] = ixyr[0] - w_pad
    pxyr[1] = pxyr[1] - h_pad
    ixyr[1] = ixyr[1] - h_pad
    
    tkimage1 = ImageTk.PhotoImage(im.resize(square_size))
    myvar1.configure(image=tkimage1)
    myvar1.image = tkimage1
    alpha = alpha_slider.get()
    ld_im = ldeformer.linear_deform(im.copy(), mask.copy(), pxyr.copy(), ixyr.copy(), alpha)
    tkimage2 = ImageTk.PhotoImage(ld_im.resize(square_size))
    myvar2.configure(image=tkimage2)
    myvar2.image = tkimage2
    bd_im = bdeformer.biomechanical_deform(im.copy(), mask.copy(), pxyr.copy(), ixyr.copy(), alpha)
    tkimage3 = ImageTk.PhotoImage(bd_im.resize(square_size))
    myvar3.configure(image=tkimage3)
    myvar3.image = tkimage3
    dnet_im = eyepreserve.deform_crop_with_alpha(im.copy(), mask.copy(), back_mask.copy(), pxyr.copy(), ixyr.copy(), alpha, device)
    tkimage4 = ImageTk.PhotoImage(dnet_im.resize(square_size))
    myvar4.configure(image=tkimage4)
    myvar4.image = tkimage4

Button(root, text='Generate New Image', command=randomizeimage).pack()

alpha_slider = Scale(root, from_=.2, to=.7, resolution=0.01, orient=HORIZONTAL)
alpha_slider.set(round(alpha, 2))
alpha_slider.pack()

def alphachange():
    global im, mask, back_mask, pxyr, ixyr, alpha
    alpha = alpha_slider.get()
    ld_im = ldeformer.linear_deform(im.copy(), mask.copy(), pxyr.copy(), ixyr.copy(), alpha)
    tkimage2 = ImageTk.PhotoImage(ld_im.resize(square_size))
    myvar2.configure(image=tkimage2)
    myvar2.image = tkimage2
    bd_im = bdeformer.biomechanical_deform(im.copy(), mask.copy(), pxyr.copy(), ixyr.copy(), alpha)
    tkimage3 = ImageTk.PhotoImage(bd_im.resize(square_size))
    myvar3.configure(image=tkimage3)
    myvar3.image = tkimage3
    dnet_im = eyepreserve.deform_crop_with_alpha(im.copy(), mask.copy(), back_mask.copy(), pxyr.copy(), ixyr.copy(), alpha, device)
    tkimage4 = ImageTk.PhotoImage(dnet_im.resize(square_size))
    myvar4.configure(image=tkimage4)
    myvar4.image = tkimage4
    
Button(root, text='Set', command=alphachange).pack()
frame.pack(expand=1)
root.mainloop()



