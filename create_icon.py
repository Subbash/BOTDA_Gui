"""
Script to create the DSS BOTDA application icon.
Run this once to generate the icon file.
"""
from PIL import Image, ImageDraw, ImageFont
import os

def create_botda_icon():
    """Create the DSS BOTDA icon."""
    # Create image with transparent background
    size = 256
    img = Image.new('RGBA', (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Draw a circular background
    circle_color = (52, 152, 219, 255)  # Blue color
    padding = 10
    draw.ellipse([padding, padding, size-padding, size-padding], fill=circle_color)

    # Draw inner circle (representing fiber optic)
    inner_padding = 40
    inner_color = (41, 128, 185, 255)  # Darker blue
    draw.ellipse([inner_padding, inner_padding, size-inner_padding, size-inner_padding], fill=inner_color)

    # Draw wave pattern (representing Brillouin scattering)
    wave_color = (236, 240, 241, 255)  # Light gray
    center = size // 2
    for i in range(3):
        radius = 60 + i * 25
        draw.arc([center-radius, center-radius, center+radius, center+radius],
                 start=0, end=180, fill=wave_color, width=4)
        draw.arc([center-radius, center-radius, center+radius, center+radius],
                 start=180, end=360, fill=wave_color, width=4)

    # Draw text "DSS"
    try:
        # Try to use a system font
        font_size = 70
        try:
            font = ImageFont.truetype("arialbd.ttf", font_size)  # Arial Bold
        except:
            try:
                font = ImageFont.truetype("Arial Bold.ttf", font_size)
            except:
                try:
                    font = ImageFont.truetype("arial.ttf", font_size)
                except:
                    font = ImageFont.load_default()
    except:
        font = ImageFont.load_default()

    text = "DSS"
    # Get text bounding box
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    text_x = (size - text_width) // 2
    text_y = (size - text_height) // 2 - 5

    # Draw text with shadow for better visibility
    shadow_offset = 2
    draw.text((text_x + shadow_offset, text_y + shadow_offset), text,
              fill=(0, 0, 0, 180), font=font)
    draw.text((text_x, text_y), text, fill=(255, 255, 255, 255), font=font)

    # Save as PNG
    png_path = os.path.join(os.path.dirname(__file__), 'dss_icon.png')
    img.save(png_path, 'PNG')
    print(f"Icon saved to: {png_path}")

    # Save as ICO (Windows icon format)
    ico_path = os.path.join(os.path.dirname(__file__), 'dss_icon.ico')
    img.save(ico_path, format='ICO', sizes=[(256, 256), (128, 128), (64, 64), (48, 48), (32, 32), (16, 16)])
    print(f"Icon saved to: {ico_path}")

    return png_path, ico_path

if __name__ == "__main__":
    try:
        png_path, ico_path = create_botda_icon()
        print("\nIcon creation successful!")
        print("The icon has been saved in both PNG and ICO formats.")
    except Exception as e:
        print(f"Error creating icon: {e}")
        print("\nNote: You need PIL/Pillow library installed.")
        print("Install it with: pip install Pillow")
