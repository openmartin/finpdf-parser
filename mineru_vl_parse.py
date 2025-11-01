from PIL import Image
from mineru_vl_utils import MinerUClient

client = MinerUClient(
    backend="http-client",
    server_url="http://127.0.0.1:8000"
)

image = Image.open("/path/to/the/test/image.png")
extracted_blocks = client.two_step_extract(image)
print(extracted_blocks)