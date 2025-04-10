import streamlit as st
import torch
import torchvision.transforms as transforms
from torchvision.models import vit_b_16
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transformer = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

@st.cache(allow_output_mutation=True)
def load_the_model():
    model = vit_b_16(pretrained=True)
    number_features = model.heads.head.in_features
    model.heads.head = torch.nn.Linear(number_features, 2)
    model.load_state_dict(torch.load("best_vit_model.pth", map_location=device))
    model = model.to(device)
    model.eval()
    return model

model = load_the_model()

st.title("Cat vs Dog Classifier")
st.write("Upload an image and the model will predict whether it is a cat or a dog.")

Uploaded_File = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if Uploaded_File is not None:
    try:
        image = Image.open(Uploaded_File).convert("RGB")
    except Exception as e:
        st.error("Error loading image!")
    
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    input_image = transformer(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(input_image)
        _, predicted = torch.max(outputs, 1)
    
    class_names = ["Cat", "Dog"]
    prediction = class_names[predicted.item()]
    
    st.write(f"### Prediction: {prediction}")
