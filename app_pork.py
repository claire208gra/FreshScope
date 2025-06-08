from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.responses import JSONResponse
import base64
import io
from PIL import Image
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2
from yolox.utils import postprocess, vis
import torch
import argparse
import os
import time
from loguru import logger

import cv2

import torch

from yolox.data.data_augment import ValTransform
from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, postprocess, vis
# Importing YOLOX
from yolox.exp import get_exp
from yolox.utils import fuse_model, postprocess

app = FastAPI()

# Cấu hình các nguồn (origins) được phép truy cập
origins = [
    "http://localhost:8807",  # URL của frontend trong môi trường phát triển
    "http://127.0.0.1:8807", # Localhost khác
    "https://porkai.phoenixtech.vn",
    "https://porkai.phoenixtech.vn:8807"  # URL chính thức của frontend
]

# Thêm CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Các domain được phép
    allow_credentials=True,  # Cho phép cookie, thông tin đăng nhập
    allow_methods=["*"],     # Các phương thức HTTP được phép (GET, POST, v.v.)
    allow_headers=["*"],     # Các header được phép
)
classes_names = ["Không rõ","thịt tươi", "thịt ướp lạnh", "thịt ôi thiu"]
# Load YOLOX model (Replace 'yolox_config' and 'model_weights' with actual paths)
class Predictor(object):
    def __init__(
        self,
        model,
        exp,
        cls_names=classes_names,
        trt_file=None,
        decoder=None,
        device="cpu",
        fp16=False,
        legacy=False,
    ):
        self.model = model
        self.cls_names = cls_names
        self.decoder = decoder
        self.num_classes = exp.num_classes
        self.confthre = 0.5
        self.nmsthre = 0.6
        self.test_size = exp.test_size
        self.device = device
        self.fp16 = fp16
        self.preproc = ValTransform(legacy=legacy)
        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones(1, 3, exp.test_size[0], exp.test_size[1]).cuda()
            self.model(x)
            self.model = model_trt

    def inference(self, img):
        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = os.path.basename(img)
            img = cv2.imread(img)
        else:
            img_info["file_name"] = None

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        ratio = min(self.test_size[0] / img.shape[0], self.test_size[1] / img.shape[1])
        img_info["ratio"] = ratio

        img, _ = self.preproc(img, None, self.test_size)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.float()
        if self.device == "gpu":
            img = img.cuda()
            if self.fp16:
                img = img.half()  # to FP16

        with torch.no_grad():
            t0 = time.time()
            outputs = self.model(img)
            if self.decoder is not None:
                outputs = self.decoder(outputs, dtype=outputs.type())
            outputs = postprocess(
                outputs, self.num_classes, self.confthre,
                self.nmsthre, class_agnostic=True
            )
            logger.info("Infer time: {:.4f}s".format(time.time() - t0))
        return outputs, img_info
    
    def post_processing(self, output, img_info, cls_conf=0.35):
        ratio = img_info["ratio"]
        img = img_info["raw_img"]

        # Check if output is valid
        if output is None or output.shape[0] == 0:
            return [], [], []

        output = output.cpu()

        bboxes = output[:, 0:4]
        bboxes /= ratio  # preprocessing: resize

        cls = output[:, 6]
        scores = output[:, 4] * output[:, 5]
        
        return bboxes, cls, scores


# Define schema for receiving image data
class ImageData(BaseModel):
    image: str  # Base64 encoded image từ client

exp = get_exp(exp_file="/home/namnh1/YOLOX/exps/example/custom/yolox_s_pork.py", exp_name=None)
model = exp.get_model()
model.cuda()
model.eval()
ckpt = torch.load("/home/namnh1/YOLOX/YOLOX_outputs/yolox_s_pork/epoch_250_ckpt.pth", map_location="cpu")
model.load_state_dict(ckpt["model"])
predictor = Predictor(model=model, exp=exp, cls_names=classes_names, trt_file=None, decoder=None, device="gpu", fp16=False, legacy=False)

COCO_CLASSES = ["Không rõ","thịt tươi", "thịt ướp lạnh", "thịt ôi thiu"]

@app.post("/predict")
async def predict(image_data: ImageData):
    try:
        # Decode Base64 image
        prefix, base64_image = image_data.image.split(",")
        image_bytes = base64.b64decode(base64_image)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Convert image to NumPy array for inference
        input_image = np.array(image)

        # Perform inference
        outputs, img_info = predictor.inference(input_image)
        bboxes, cls_ids, scores = predictor.post_processing(outputs[0], img_info, predictor.confthre)
        print(bboxes, cls_ids, scores)
        
        # if not bboxes:
        #     return JSONResponse(content={"boxes": []})
        # Parse YOLOX outputs
        detections = []
        for i in range(len(bboxes)):
            box = bboxes[i]
            cls_id = int(cls_ids[i])
            score = scores[i]
            if score < 0.5:
                continue
            x0 = int(box[0])
            y0 = int(box[1])
            x1 = int(box[2])
            y1 = int(box[3])
            label_name = COCO_CLASSES[int(cls_id)]
            detections.append({
                "x": int(x0),
                "y": int(y0),
                "width": int((x1 - x0)),
                "height": int((y1 - y0)),
                "label": label_name,
                "confidence": round(float(score) * 100, 2),
            })
            
        print(detections)
            
        img = img_info["raw_img"]
        height, width = img.shape[:2]
        return JSONResponse(content={"boxes": detections,"input_width": width, "input_height": height})

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "test:app", 
        host="0.0.0.0", 
        port=8710, 
        ssl_keyfile="/home/staging/deploy/dxtech.vn.key", 
        ssl_certfile="/home/staging/deploy/dxtech.vn.crt"
    )
