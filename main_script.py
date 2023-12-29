import cv2
import numpy as np
import torch
import PIL.Image
import PIL.ImageDraw
import torchvision.transforms as transforms
from mit_semseg.models import ModelBuilder, SegmentationModule
from mit_semseg.utils import colorEncode
import scipy.io
import os
import csv

# Clear any cached data in GPU
torch.cuda.empty_cache()

# Import model components from the semantic segmentation library
from mit_semseg.models import ModelBuilder, SegmentationModule
from mit_semseg.utils import colorEncode
# Load the semantic segmentation model components
net_encoder = ModelBuilder.build_encoder(
    arch='resnet50dilated',
    fc_dim=2048,
    weights='encoder_epoch.pth')
net_decoder = ModelBuilder.build_decoder(
    arch='ppm_deepsup',
    fc_dim=2048,
    num_class=150,
    weights='decoder_epoch.pth',
    use_softmax=True)

crit = torch.nn.NLLLoss(ignore_index=-1)
segmentation_module = SegmentationModule(net_encoder, net_decoder, crit)
segmentation_module.eval()
segmentation_module.cuda()

# Load colors and names for visualization
colors = scipy.io.loadmat('data/color150.mat')['colors']

# Function to visualize segmentation results
def visualize_result(img, pred):
    pred_color = colorEncode(pred, colors).astype(np.uint8)
    im_vis = np.concatenate((img, pred_color), axis=1)
    return PIL.Image.fromarray(im_vis)

# Transform for input images
pil_to_tensor = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Create an output directory if it doesn't exist
output_dir = "output"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
# Load colors and names for visualization
colors = scipy.io.loadmat('data/color150.mat')['colors']

# Load class names
names = {}
with open('data/object150_info.csv') as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        names[int(row[0]) - 1] = row[5].split(";")[0]  # Adjust the index

# Function to visualize specific segmentation results
def visualize_class(img, pred, class_index, class_name):
    pred_class = pred.copy()
    pred_class[pred_class != class_index] = -1
    pred_color = colorEncode(pred_class, colors).astype(np.uint8)
    im_vis = np.concatenate((img, pred_color), axis=1)
    im_vis_pil = PIL.Image.fromarray(im_vis)
    draw = PIL.ImageDraw.Draw(im_vis_pil)
    draw.text((20, 20), class_name, fill=(255, 255, 255))
    return im_vis_pil


# Initialize the camera
cam = cv2.VideoCapture(0)

# Variables to store the modes and class index
realtime_mode = True
type_visualization = False
current_class_index = 0

while True:
    ret, frame = cam.read()
    if not ret:
        break
    # Convert frame to PIL Image
    cv2_im = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_im = PIL.Image.fromarray(cv2_im)

    # Preprocess image
    img_data = pil_to_tensor(pil_im)
    singleton_batch = {'img_data': img_data[None].cuda()}
    output_size = img_data.shape[1:]

    # Handling key inputs
    k = cv2.waitKey(1) & 0xFF
    if k == 27:  # Esc key
        break
    elif k == ord(' '):  # Space key
        realtime_mode = not realtime_mode
    elif k == 9:  # Shift key
        type_visualization = not type_visualization
    elif k >= ord('1') and k <= ord('9'):
        current_class_index = k - ord('1')
    elif k >= ord('a') and k <= ord('f'):
        current_class_index = k - ord('a') + 9

    if realtime_mode or k == 13:  # Enter key for non-realtime mode
        # Perform segmentation
        with torch.no_grad():
            scores = segmentation_module(singleton_batch, segSize=output_size)
        _, pred = torch.max(scores, dim=1)
        pred = pred.cpu()[0].numpy()

        # Determine the most common classes
        predicted_classes = np.bincount(pred.flatten()).argsort()[::-1]

        # Visualize result
        if type_visualization and current_class_index < len(predicted_classes):
            class_index = predicted_classes[current_class_index]
            class_name = names.get(class_index, "Unknown")
            vis_im = visualize_class(cv2_im, pred, class_index, class_name)
            vis_im = np.array(vis_im)
            cv2.imshow("Semantic Segmentation", cv2.cvtColor(vis_im, cv2.COLOR_RGB2BGR))
        else:
            vis_im = visualize_result(cv2_im, pred)
            vis_im = np.array(vis_im)
            cv2.imshow("Semantic Segmentation", cv2.cvtColor(vis_im, cv2.COLOR_RGB2BGR))
    else:
        # Display the live camera feed in non-realtime mode
        cv2.imshow("Camera Feed", frame)

    # Save the current frame and segmentation with 's' key
    if k == ord('s'):
        cv2.imwrite("captured_frame.jpg", frame)
        if 'vis_im' in locals():
            PIL.Image.fromarray(vis_im).save("segmented_frame.jpg")
        print("Frame and segmentation result saved.")

# Release the camera and destroy all windows
cam.release()
cv2.destroyAllWindows()