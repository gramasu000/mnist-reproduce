from PIL import Image

_ROWS = 28
_COLS = 28

def fast_normalize(image_input):
    return image_input / 255

def convert_to_image(image_input, index):
    image_array = image_input[index, :]
    image_array = np.reshape(image_array, (_ROWS, _COLS))
    image = Image.fromarray(image_array, mode="L")
    image.save(f"assets/image_{index}.jpg", "JPEG")
