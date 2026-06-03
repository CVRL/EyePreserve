from LinearDeformer import *
from BiomechDeformer import *
from CircleMaskFinder import *
from EyePreserve import *
from SyntheticIrisGenerator import *
from tkinter import Frame, Button, Scale, Label, Tk, HORIZONTAL, StringVar, OptionMenu, ttk
from PIL import ImageTk
import threading
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

root = Tk()
root.title("Synthetic Iris Synthesis and Deformation")

circlemaskfinder = CircleMaskFinder(mask_net_path="./models/nestedsharedatrousresunet-156-0.026496-maskIoU-0.943222.pth", circle_net_path='./models/convnext_tiny-1076-0.030622-maskIoU-0.938355.pth', eyelid_net_path="./models/nestedsharedatrousresunet-1935-0.036982-maskIoU-0.946174-insideeyelid.pth", device=device)
ldeformer = LinearDeformer(device=device)
bdeformer = BiomechDeformer(device=device)
eyepreserve = EyePreserve(net_path="./models/0007-val_loss-22.957276336365826+-0.0-val_bit_match-81.10588183031244+-0.0-val_linear_bit_match-75.13033321984577+-0.0.pth", device=device)
generator = SyntheticIrisGenerator(device=device)

# Note: This initial generation at startup is synchronous and will take a moment 
# before the UI opens. 
generator.generate_image(model_type='gan') 
im = generator.generate_image(model_type='diffusion').convert('L')

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

modelvar3=Label(frame, text="EyePreserve", font=('Arial', 20))
modelvar3.grid(row=0, column=3)

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

# --- THREADING AND PROGRESS BAR LOGIC ---
@torch.inference_mode()
def randomizeimage():
    # Disable controls so user doesn't interrupt the running process
    generate_btn.config(state="disabled")
    set_btn.config(state="disabled")
    
    # Show and start indeterminate progress bar
    progress.pack(side="left", padx=10)
    progress.start(10)
    
    # Send the heavy workload to a background thread
    threading.Thread(target=_generate_thread, daemon=True).start()

def _generate_thread():
    # 1. Heavy computation (Pipeline, Deformation) runs here entirely off the main thread.
    selected_model = dropdown_var.get()
    type_arg = 'gan' if selected_model == "StyleGAN3" else 'diffusion'
    
    new_im = generator.generate_image(model_type=type_arg).convert("L")
    
    ow, oh = new_im.size
    new_im, w_pad, h_pad = circlemaskfinder.fix_image(new_im)
    new_mask, new_back_mask, new_pxyr, new_ixyr = circlemaskfinder.segmentAndCircApprox(new_im.convert("L"))
    w, h = new_im.size
    new_im = new_im.crop((w_pad, h_pad, w_pad + ow, h_pad + oh))
    new_mask = new_mask.crop((w_pad, h_pad, w_pad + ow, h_pad + oh))
    new_back_mask = new_back_mask.crop((w_pad, h_pad, w_pad + ow, h_pad + oh))
    new_pxyr[0] -= w_pad
    new_ixyr[0] -= w_pad
    new_pxyr[1] -= h_pad
    new_ixyr[1] -= h_pad
    
    current_alpha = alpha_slider.get()
    
    new_ld_im = ldeformer.linear_deform(new_im.copy(), new_mask.copy(), new_pxyr.copy(), new_ixyr.copy(), current_alpha)
    new_bd_im = bdeformer.biomechanical_deform(new_im.copy(), new_mask.copy(), new_pxyr.copy(), new_ixyr.copy(), current_alpha)
    new_dnet_im = eyepreserve.deform_crop_with_alpha(new_im.copy(), new_mask.copy(), new_back_mask.copy(), new_pxyr.copy(), new_ixyr.copy(), current_alpha, device)

    # 2. Package all results and schedule UI update on the MAIN thread safely
    root.after(0, _update_ui, new_im, new_mask, new_back_mask, new_pxyr, new_ixyr, current_alpha, new_ld_im, new_bd_im, new_dnet_im)

def _update_ui(new_im, new_mask, new_back_mask, new_pxyr, new_ixyr, current_alpha, new_ld_im, new_bd_im, new_dnet_im):
    global im, mask, back_mask, pxyr, ixyr, alpha
    
    # Set the globals to the newly generated data
    im, mask, back_mask = new_im, new_mask, new_back_mask
    pxyr, ixyr, alpha = new_pxyr, new_ixyr, current_alpha
    
    # Update UI Photos (ImageTk MUST be updated on the main thread)
    tkimg1 = ImageTk.PhotoImage(im.resize(square_size))
    myvar1.configure(image=tkimg1)
    myvar1.image = tkimg1
    
    tkimg2 = ImageTk.PhotoImage(new_ld_im.resize(square_size))
    myvar2.configure(image=tkimg2)
    myvar2.image = tkimg2
    
    tkimg3 = ImageTk.PhotoImage(new_bd_im.resize(square_size))
    myvar3.configure(image=tkimg3)
    myvar3.image = tkimg3
    
    tkimg4 = ImageTk.PhotoImage(new_dnet_im.resize(square_size))
    myvar4.configure(image=tkimg4)
    myvar4.image = tkimg4
    
    # Stop progress bar, hide it, and re-enable buttons
    progress.stop()
    progress.pack_forget()
    generate_btn.config(state="normal")
    set_btn.config(state="normal")
# ----------------------------------------

# --- New Control Frame for Side-by-Side UI Elements ---
control_frame = Frame(root)
control_frame.pack(pady=10) 

dropdown_var = StringVar(root)
dropdown_var.set("DDPM") # Default value

model_dropdown = OptionMenu(control_frame, dropdown_var, "DDPM", "StyleGAN3")
model_dropdown.pack(side="left", padx=5)

generate_btn = Button(control_frame, text='Generate New Image', command=randomizeimage)
generate_btn.pack(side="left", padx=5)

# Initialize progress bar (we pack and unpack this dynamically)
progress = ttk.Progressbar(control_frame, orient="horizontal", length=200, mode="indeterminate")
# --------------------------------------------------------

alpha_slider = Scale(root, from_=.2, to=.7, resolution=0.01, orient=HORIZONTAL)
alpha_slider.set(round(alpha, 2))
alpha_slider.pack()

@torch.inference_mode()
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

set_btn = Button(root, text='Set', command=alphachange)
set_btn.pack(pady=5)

frame.pack(expand=1)
root.mainloop()