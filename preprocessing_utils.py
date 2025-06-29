
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def get_char_mapping():
    """Create mapping from class labels to characters"""
    chars = []
    for i in range(10):
        chars.append(str(i))
    for i in range(26):
        chars.append(chr(ord('A') + i))
    for i in range(26):
        chars.append(chr(ord('a') + i))
    return chars

def advanced_preprocess_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Could not read image")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)

    filtered = cv2.bilateralFilter(enhanced, 9, 75, 75)

    binary = cv2.adaptiveThreshold(
        filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )

    kernel = np.ones((2,2), np.uint8)
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)

    return cleaned

def improved_character_extraction(image, bbox):
    x, y, w, h = bbox

    pad = 5
    x_start = max(0, x - pad)
    y_start = max(0, y - pad)
    x_end = min(image.shape[1], x + w + pad)
    y_end = min(image.shape[0], y + h + pad)

    char_img = image[y_start:y_end, x_start:x_end]

    aspect_ratio = w / h

    if aspect_ratio > 1:
        new_w = 20
        new_h = int(20 / aspect_ratio)
    else:
        new_h = 20
        new_w = int(20 * aspect_ratio)

    new_w = max(new_w, 8)
    new_h = max(new_h, 8)

    resized = cv2.resize(char_img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    final_img = np.zeros((28, 28), dtype=np.uint8)
    start_x = (28 - new_w) // 2
    start_y = (28 - new_h) // 2
    final_img[start_y:start_y+new_h, start_x:start_x+new_w] = resized

    final_img = cv2.GaussianBlur(final_img, (3, 3), 0.5)

    normalized = final_img.astype('float32') / 255.0

    return normalized

def segment_characters_with_spaces(binary_image):
    contours, _ = cv2.findContours(
        binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    character_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 100:
            x, y, w, h = cv2.boundingRect(contour)
            if 0.2 < h/w < 5 and w > 10 and h > 10:
                character_contours.append((x, y, w, h))

    character_contours.sort(key=lambda x: x[0])

    if not character_contours:
        return []

    segments_with_spaces = []

    for i, bbox in enumerate(character_contours):
        x, y, w, h = bbox

        segments_with_spaces.append(('char', bbox))

        if i < len(character_contours) - 1:
            next_bbox = character_contours[i + 1]
            next_x = next_bbox[0]

            current_end = x + w
            distance = next_x - current_end

            avg_char_width = np.mean([bbox[2] for bbox in character_contours])

            space_threshold = avg_char_width * 0.8

            if distance > space_threshold:
                segments_with_spaces.append(('space', None))

    return segments_with_spaces

def predict_with_confidence_analysis(image_path, model, char_mapping, confidence_threshold=0.7, show_plots=False):
    """Enhanced prediction with confidence analysis - modified for Flask deployment"""
    try:
        binary_img = advanced_preprocess_image(image_path)

        segments = segment_characters_with_spaces(binary_img)

        if not segments:
            print("No characters detected in the image")
            return ""

        predictions = []
        low_confidence_chars = []
        char_position = 0

        actual_chars = [seg for seg in segments if seg[0] == 'char']

        # Only show plots if explicitly requested (not recommended for Flask)
        if show_plots and actual_chars:
            plt.figure(figsize=(20, 8))

        for i, (seg_type, bbox) in enumerate(segments):
            if seg_type == 'space':
                predictions.append((' ', 1.0, [' '], [1.0]))
                if show_plots:
                    print(f"  Space detected at position {i}")
            else:
                char_img = improved_character_extraction(binary_img, bbox)

                char_input = char_img.reshape(1, 28, 28, 1)
                prediction = model.predict(char_input, verbose=0)

                top_3_indices = np.argsort(prediction[0])[-3:][::-1]
                top_3_probs = prediction[0][top_3_indices]
                top_3_chars = [char_mapping[idx] for idx in top_3_indices]

                predicted_char = top_3_chars[0]
                confidence = top_3_probs[0]

                predictions.append((predicted_char, confidence, top_3_chars, top_3_probs))

                if confidence < confidence_threshold:
                    low_confidence_chars.append({
                        'position': i,
                        'char_position': char_position,
                        'char': predicted_char,
                        'confidence': confidence,
                        'alternatives': list(zip(top_3_chars, top_3_probs))
                    })

                # Only create plots if show_plots is True
                if show_plots:
                    plt.subplot(3, len(actual_chars), char_position + 1)
                    plt.imshow(char_img, cmap='gray')
                    if confidence < confidence_threshold:
                        plt.title(f'{predicted_char}\n{confidence:.2f}', color='red')
                    else:
                        plt.title(f'{predicted_char}\n{confidence:.2f}')
                    plt.axis('off')

                    plt.subplot(3, len(actual_chars), len(actual_chars) + char_position + 1)
                    y_pos = np.arange(3)
                    plt.barh(y_pos, top_3_probs)
                    plt.yticks(y_pos, top_3_chars)
                    plt.xlabel('Confidence')
                    plt.title('Top 3')

                    plt.subplot(3, len(actual_chars), 2*len(actual_chars) + char_position + 1)
                    x, y, w, h = bbox
                    original_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                    char_region = original_img[y:y+h, x:x+w]
                    plt.imshow(char_region, cmap='gray')
                    plt.title('Original')
                    plt.axis('off')

                char_position += 1

        if show_plots and actual_chars:
            plt.tight_layout()
            plt.show()

        predicted_text = ''.join([pred[0] for pred in predictions])
        char_confidences = [pred[1] for pred in predictions if pred[0] != ' ']
        avg_confidence = np.mean(char_confidences) if char_confidences else 0

        if show_plots:
            print(f"Predicted text: '{predicted_text}'")
            print(f"Average confidence: {avg_confidence:.3f}")
            print(f"Total segments: {len(segments)} (Characters: {len(actual_chars)}, Spaces: {len(segments) - len(actual_chars)})")

            if low_confidence_chars:
                print(f"\n⚠️  LOW CONFIDENCE CHARACTERS ({len(low_confidence_chars)}):")
                for char_info in low_confidence_chars:
                    pos = char_info['position']
                    char_pos = char_info['char_position']
                    char = char_info['char']
                    conf = char_info['confidence']
                    alts = char_info['alternatives']

                    print(f"  Position {pos} (Char #{char_pos}): '{char}' (confidence: {conf:.2f})")
                    print(f"    Alternatives: {', '.join([f'{c}({p:.2f})' for c, p in alts[1:]])}")

                    if char in ['M', 'm', 'w', 'W']:
                        print(f"    💡 M/m/w/W confusion detected. Check letter shape and size.")
                    elif char in ['s', 'S', '5']:
                        print(f"    💡 s/S/5 confusion detected. Check curves and angles.")
                    elif char in ['o', 'O', '0']:
                        print(f"    💡 o/O/0 confusion detected. Check size and thickness.")
                    elif char in ['I', 'l', '1', 'T']:
                        print(f"    💡 I/l/1/T confusion detected. Check serifs and horizontal lines.")

        return predicted_text, predictions, low_confidence_chars

    except Exception as e:
        print(f"Error processing image: {str(e)}")
        return ""
