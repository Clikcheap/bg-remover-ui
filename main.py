from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from rembg import remove, new_session
from PIL import Image, ImageEnhance, ImageFilter
import io

app = FastAPI()

# Use high-quality model
session = new_session(model_name="isnet-general-use")

@app.post("/remove-bg")
async def remove_bg(
    file: UploadFile = File(...),
    alpha_matting: bool = Form(False)
):
    input_image = await file.read()

    # Pre-process image for better edge detection
    img = Image.open(io.BytesIO(input_image)).convert("RGB")
    img.thumbnail((1024, 1024))  # Resize up to 1024px max
    img = img.filter(ImageFilter.MedianFilter(size=3))  # Denoise (optional)
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(1.8)  # Boost contrast

    # Convert to bytes
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    buffer.seek(0)

    # Remove background with optional alpha matting
    output_image = remove(
        buffer.read(),
        session=session,
        alpha_matting=alpha_matting,
        alpha_matting_foreground_threshold=240,
        alpha_matting_background_threshold=10,
        alpha_matting_erode_size=10
    )

    # Post-process: crop transparent edges and resize to max 800px
    result_img = Image.open(io.BytesIO(output_image)).convert("RGBA")
    result_img = result_img.crop(result_img.getbbox())
    result_img.thumbnail((800, 800))
    result_io = io.BytesIO()
    result_img.save(result_io, format="PNG")
    result_io.seek(0)

    return StreamingResponse(result_io, media_type="image/png")

# Serve your front-end
app.mount("/", StaticFiles(directory="static", html=True), name="static")

