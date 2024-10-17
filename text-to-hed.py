import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
from comfy.model_nodes import NODE_CLASS_MAPPINGS, custom_node

# Hàm tạo văn bản ra ảnh
def create_text_image(text, font_path, font_size, image_size):
    # Tạo ảnh trắng
    image = Image.new("RGB", image_size, "white")
    draw = ImageDraw.Draw(image)
    
    # Tải font
    font = ImageFont.truetype(font_path, font_size)
    
    # Tính toán vị trí để căn giữa văn bản
    text_width, text_height = draw.textsize(text, font=font)
    position = ((image_size[0] - text_width) // 2, (image_size[1] - text_height) // 2)
    
    # Vẽ văn bản
    draw.text(position, text, font=font, fill="black")
    
    return image

# Hàm áp dụng HED
def apply_hed(image):
    # Chuyển đổi ảnh Pillow sang numpy array
    image = np.array(image)
    
    # Chuyển sang grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Áp dụng Canny Edge Detection
    edges = cv2.Canny(gray, 100, 200)
    
    return edges

# Custom Node trong ComfyUI
class TextToHEDNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"default": "VIỆT NAM", "multiline": False}),
                "font_path": ("STRING", {"default": "path_to_your_bold_font.ttf", "multiline": False}),
                "font_size": ("INT", {"default": 150, "min": 10, "max": 500}),
                "width": ("INT", {"default": 500, "min": 100, "max": 2000}),
                "height": ("INT", {"default": 300, "min": 100, "max": 2000}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate_hed_image"
    CATEGORY = "Image Generation"

    def generate_hed_image(self, text, font_path, font_size, width, height):
        # Tạo ảnh văn bản
        image = create_text_image(text, font_path, font_size, (width, height))
        
        # Áp dụng HED
        hed_image = apply_hed(image)
        
        # Chuyển numpy array trở lại dạng ảnh PIL
        hed_image_pil = Image.fromarray(hed_image)
        
        return (hed_image_pil,)

# Đăng ký custom node vào hệ thống
NODE_CLASS_MAPPINGS["TextToHED"] = TextToHEDNode
