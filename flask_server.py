from flask import Flask, request, jsonify
import os
from infer import resize_and_save, create_patches, extract_features, add_nearest, merge_h5, infer_and_get_attention

app = Flask(__name__)
UPLOAD_FOLDER = "./uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/infer', methods=['POST'])
def infer_endpoint():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    filename = file.filename
    slide_id = os.path.splitext(filename)[0]

    save_path = os.path.join(UPLOAD_FOLDER, filename)
    resized_path = os.path.join(UPLOAD_FOLDER, f"{slide_id}_resized.jpg")
    patch_dir = os.path.join(UPLOAD_FOLDER, f"patches_{slide_id}")
    h5_path = os.path.join(UPLOAD_FOLDER, f"{slide_id}.h5")
    final_h5_path = os.path.join(UPLOAD_FOLDER, f"{slide_id}_final.h5")

    try:
        file.save(save_path)
        resize_and_save(save_path, resized_path)
        create_patches(resized_path, patch_dir)
        extract_features(patch_dir, h5_path)
        add_nearest(h5_path)
        merge_h5(h5_path, final_h5_path)
        pred_class, softmax_probs, attention_base64 = infer_and_get_attention(final_h5_path, resized_path)

        return jsonify({
            "pred_class": int(pred_class),
            "softmax_probs": softmax_probs,
            "attention_map_base64": attention_base64
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
