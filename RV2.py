import cv2
import numpy as np
import math
import onnxruntime

def load_onnx_model(modelPath):
    return modelPath
    
def sam(onnxruntime_sam_encoder, onnxruntime_sam_decoder, image, input_point, input_label):
    # Convert input_point (tuple) to NumPy array
    np_input_point = np.array(input_point, dtype=np.float32)[None, :]  # Shape (1, 2)
    np_input_label =  np.array([input_label])
    orig_height, orig_width = image.shape[:2]

    def prepare_inputs_encoder(image, ort_session):
        # Preprocess the image and convert it into a blob
        image = cv2.resize(image, (1024, 1024))
        blob = cv2.dnn.blobFromImage(image, 1/256)
        # Prepare the inputs for the encoder model
        inputs_encoder = {ort_session.get_inputs()[0].name: blob}
        return inputs_encoder

    def prepare_decoder_inputs(image_embedding, input_point, input_label):
        # Prepare the coordinates for the input points, adding a dummy point
        onnx_coord = np.concatenate([input_point, np.array([[0.0, 0.0]])], axis=0)[None, :, :].astype(np.float32)
        # Prepare the labels for the input points, adding a dummy label
        onnx_label = np.concatenate([input_label, np.array([-1])], axis=0)[None, :].astype(np.float32)
        # Prepare the inputs for the decoder model
        ort_inputs_decoder = {
            "image_embeddings": image_embedding.astype(np.float32),
            "point_coords": onnx_coord,
            "point_labels": onnx_label,
        }
        return ort_inputs_decoder

    # Prepare the inputs for the encoder
    inputs_encoder = prepare_inputs_encoder(image, onnxruntime_sam_encoder)
    result_encoder = onnxruntime_sam_encoder.run(None, inputs_encoder)
    image_embedding = np.array(result_encoder[0])

    # Adjust input_point according to image resizing
    scale_x = 1024 / orig_width
    scale_y = 1024 / orig_height
    np_input_point_scaled = np_input_point * np.array([scale_x, scale_y])

    # Decoder inference
    ort_inputs_decoder = prepare_decoder_inputs(image_embedding, np_input_point_scaled, np_input_label)
    result_decoder = onnxruntime_sam_decoder.run(None, ort_inputs_decoder)
    low_res_logits, maskss = result_decoder

    # Apply a binary threshold to convert the mask probabilities to a binary mask
    _, binaryMasks = cv2.threshold(maskss[0][2], 0, 255, cv2.THRESH_BINARY)
    # Resize the mask to match the original image dimensions
    binaryMasks = cv2.resize(binaryMasks, (image.shape[1], image.shape[0]))
    
    return binaryMasks, {"tst" : 12, "g": np.float32(1.2)}, [5,6,7], np.array([9,10,11,12]), {'pp1' : np.array([9,10,11,12.2], dtype=np.float64), "pp2" : {"nes" : np.str_("oh my")}}
