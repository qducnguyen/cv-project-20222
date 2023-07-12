import gradio as gr
from utils import str2bool
import argparse
import glob
import cv2

from Bicubic.solver import BicubicInferencer
from SRCNN.solver import SRCNNInferencer
from VDSR.solver import VDSRInferencer
from EDSR.solver import EDSRInferencer
from SRGAN.solver import SRGANInferencer
from UNet.solver import UNetInferencer

import random
import traceback

MIN_CROP_SIZE = 25
MAX_CROP_SIZE = 250
MAX_INPUT_SIZE = 1073

size = ["x2", "x3", "x4"]
method = ["Bicubic", "SRCNN", "VDSR", "EDSR", "SRGAN", "SRU-NET"]
css = """
        #status {text-align: center !important}
        .label_img {text-align: center !important}
        """
image_names = ['0184', '0239', '0360', '0459', '0774']
images = {f"Image {i+1}":image_names[i] for i in range(len(image_names))}


def get_example(img_name, scale_factor, x, y, crop_size):
    scale_factor = int(scale_factor.replace("x", ""))
    img_name = images[img_name]
    x, y, crop_size = int(x), int(y), int(crop_size)
    images_path = glob.glob(f'image/sr/{img_name}x{scale_factor}/*.png')
    images_path.sort()

    lr_path, hr_path = f'image/examples/{img_name}x{scale_factor}.png', f"image/examples/{img_name}.png"

    # print(sorted(images_path))
    org_hr = cv2.cvtColor(cv2.imread(hr_path), cv2.COLOR_BGR2RGB)
    lr_crop_size = crop_size//scale_factor
    org_lr = cv2.cvtColor(cv2.imread(lr_path), cv2.COLOR_BGR2RGB)[x//scale_factor:lr_crop_size + x//scale_factor, y//scale_factor:lr_crop_size + y//scale_factor]

    hr = org_hr.copy()[x:x+crop_size, y:y+crop_size]
    start_point = (y, x)
    end_point = (y+crop_size, x+crop_size)
    color = (255, 0, 0)
    thickness = org_hr.shape[1]//70

    org_hr = cv2.rectangle(org_hr, start_point, end_point, color, thickness)

    image_datas = [org_hr, org_lr] + [cv2.cvtColor(cv2.imread(i), cv2.COLOR_BGR2RGB)[x:x+crop_size, y:y+crop_size] for i in images_path] + [hr]

    return image_datas


default_images = get_example("Image 1", "x4", "700", "550", "150")
# print(len(default_images))

def generate(input, m, s, progress=gr.Progress(track_tqdm=True)):
    try:
        parser = argparse.ArgumentParser(description='RUSH20222 Super-resolution Inferencer')

        # model configuration
        parser.add_argument('--model', '-m',  type=str, default="bicubic", help="model")
        parser.add_argument("--ckp_dir", type=str, default="./ckp/")
        parser.add_argument('--scale', '-s',  type=int, default=4, help="Super Resolution upscale factor")
        parser.add_argument("--image_input_path", "-in", type=str, default="examples/sample_inference_01.jpg")
        parser.add_argument("--image_output_path", "-out", type=str, default="examples/sample_inference_01_test.png")
        parser.add_argument('--attention', '-a', type=str2bool, default=True, help="Attention or Not, skip for bicubic")
        parser.add_argument("--metric", type=str2bool, default=False)
        # hyper-parameters
        args, unknown = parser.parse_known_args()

        args.image_input_path = input
        args.scale = int(s.replace('x', ''))
        args.model = m.lower()
        args.image_output_path = f'image/sr/test-{m.lower()}-{s}.png'
        if args.model == "bicubic":
            inferencer  = BicubicInferencer(args)
        elif args.model == "srcnn":
            args.attention = True
            inferencer = SRCNNInferencer(args)
        elif args.model == "vdsr":
            inferencer = VDSRInferencer(args)
        elif args.model == "edsr":
            inferencer = EDSRInferencer(args)
        elif args.model == "srgan":
            inferencer = SRGANInferencer(args)
        elif args.model == "sru-net":
            inferencer = UNetInferencer(args)
        else:
            print("No model found!")
        inferencer.run()

        # return gr.Dropdown.update(choices=inputs), "Complete!!! Please check the result tab"
        return gr.update(value=args.image_output_path), gr.update(interactive=True), "Complete!!! Check the result below"
    except Exception as e:
        print("ERROR OCCURED:", traceback.format_exc())
        return gr.update(value=None), gr.update(interactive=True), "Please choose an image"
    

def update_params(img_name):
    img_name = images[img_name]
    path = f"image/examples/{img_name}.png"
    cur_img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    return gr.update(min_width=0, maximum=cur_img.shape[0]-MIN_CROP_SIZE), gr.update(min_width=0, maximum=cur_img.shape[1]-MIN_CROP_SIZE), gr.update(min_width=MIN_CROP_SIZE, maximum=MAX_CROP_SIZE)
    

def update_crop_size(img_name, x, y):
    img_name = images[img_name]
    path = f"image/examples/{img_name}.png"
    cur_img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

    new_max = min([cur_img.shape[0] - int(x), cur_img.shape[1] - int(y)])
    new_value = value=MIN_CROP_SIZE if new_max < 150 else 150
    return gr.update(min_width=MIN_CROP_SIZE, maximum=new_max, value=new_value)

def check_input_size(input):
    try:
        cur_img = cv2.cvtColor(cv2.imread(input), cv2.COLOR_BGR2RGB)

        max_size = max([cur_img.shape[0], cur_img.shape[1]])

        if max_size < MAX_INPUT_SIZE:
            return gr.update(value="Valid input! You can continue generating SR image"), gr.update(interactive=True)
        else:
            return gr.update(value="Invliad input! Your image may exceed our limit size (1073), please change you input!!"), gr.update(interactive=False)
    except:
        return gr.update(value="Upload the image you want to test"), gr.update(interactive=True)

with gr.Blocks(css=css, theme=gr.themes.Soft(), title="Image Super Resolution") as app:
    
    with gr.Tab("Examples"):
        with gr.Row():
            img_selection = gr.Dropdown(choices=list(images.keys()),value=list(images.keys())[0], label="Image", elem_classes="dropdown")
            size_selection = gr.Dropdown(choices=size, value=size[0], label="SR size", elem_classes="dropdown")
            
            y = gr.Slider(min_width=0, maximum=default_images[0].shape[1]-MIN_CROP_SIZE, value=550, label="X", step=5)
            x = gr.Slider(min_width=0, maximum=default_images[0].shape[0]-MIN_CROP_SIZE, value=700, label="Y", step=5)
            
            r = gr.Slider(min_width=MIN_CROP_SIZE, maximum=MAX_CROP_SIZE, value=150, label="Crop size")
        with gr.Row():
            with gr.Column(scale=4):
                # gr.Markdown("Full image")
                org_hr = gr.Image(value=default_images[0], show_label=False)
            with gr.Column(scale=6):
                with gr.Row():
                    with gr.Column(min_width=70):
                        lr = gr.Image(value=default_images[1], show_label=False)
                        gr.Markdown("LR", elem_classes="label_img")
                    with gr.Column(min_width=70):
                        vdsr = gr.Image(value=default_images[7], show_label=False)
                        gr.Markdown("VDSR", elem_classes="label_img")
                    with gr.Column(min_width=70):
                        unet = gr.Image(value=default_images[6], show_label=False)
                        gr.Markdown("SRU-NET", elem_classes="label_img")
                    with gr.Column(min_width=70):
                        bicubic = gr.Image(value=default_images[2], show_label=False)
                        gr.Markdown("Bicubic", elem_classes="label_img")
                with gr.Row():
                    with gr.Column(min_width=70):
                        hr = gr.Image(value=default_images[8], show_label=False)
                        gr.Markdown("HR", elem_classes="label_img")
                    with gr.Column(min_width=70):
                        edsr = gr.Image(value=default_images[3], show_label=False)
                        gr.Markdown("EDSR", elem_classes="label_img")
                    with gr.Column(min_width=70):
                        srcnn = gr.Image(value=default_images[4], show_label=False)
                        gr.Markdown("SRCNN", elem_classes="label_img")
                    with gr.Column(min_width=70):
                        srgan = gr.Image(value=default_images[5], show_label=False)
                        gr.Markdown("SRGAN", elem_classes="label_img")

        # gr.Markdown("Examples")
        # gr.Examples(examples=[["Image 0", "x2", "550", "500", "150"]], 
        #             inputs=[img_selection, size_selection, x, y, r],
        #             fn=get_example,
        #             outputs=[org_hr, lr, bicubic, edsr, srcnn, srgan, unet, vdsr, hr],
        #             cache_examples=True)
                    
    with gr.Tab("Prediction"):
        with gr.Row():
            input = gr.Image(type="filepath")
            with gr.Column():
                method_selection_test = gr.Dropdown(choices=method, label="Method", value=method[0], interactive=True, elem_classes="dropdown")
                size_selection_test = gr.Dropdown(choices=size, label="SR size", value=size[0], interactive=True, elem_classes="dropdown")
        with gr.Row():
            submit = gr.Button(value="Submit")
        with gr.Row():
            status = gr.Markdown("Upload the image you want to test", elem_id="status")
        with gr.Row():
            output_test = gr.Image(type="filepath")
            
        
        input.change(check_input_size, inputs=input, outputs=[status, submit])
        submit.click(lambda x: gr.update(interactive=False), inputs=[submit], outputs=[submit]).then(generate, inputs=[input, method_selection_test, size_selection_test], outputs=[output_test, submit, status])

        img_selection.change(update_params, inputs=[img_selection], outputs=[x, y, r]).then(get_example, inputs=[img_selection, size_selection, x, y, r], outputs=[org_hr, lr, bicubic, edsr, srcnn, srgan, unet, vdsr, hr])
        size_selection.change(get_example, inputs=[img_selection, size_selection, x, y, r], outputs=[org_hr, lr, bicubic, edsr, srcnn, srgan, unet, vdsr, hr])
        
        x.release(update_crop_size, inputs=[img_selection, x, y], outputs=r).then(get_example, inputs=[img_selection, size_selection, x, y, r], outputs=[org_hr, lr, bicubic, edsr, srcnn, srgan, unet, vdsr, hr])
        y.release(update_crop_size, inputs=[img_selection, x, y], outputs=r).then(get_example, inputs=[img_selection, size_selection, x, y, r], outputs=[org_hr, lr, bicubic, edsr, srcnn, srgan, unet, vdsr, hr])
        r.release(get_example, inputs=[img_selection, size_selection, x, y, r], outputs=[org_hr, lr, bicubic, edsr, srcnn, srgan, unet, vdsr, hr])
        

app.queue(concurrency_count=3)
app.launch()

