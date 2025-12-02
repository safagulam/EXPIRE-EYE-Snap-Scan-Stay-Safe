import os
import io
import re
import datetime
import base64
import json
import numpy as np
import cv2
from PIL import Image
from ultralytics import YOLO
import time
# --- NEW: EasyOCR Imports ---
try:
    import easyocr
    import torch
except ImportError:
    easyocr = None
    torch = None
# ---

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager

# --- Configuration ---
# Set the time limit for classification (e.g., 30 days)
EXPIRATION_THRESHOLD_DAYS = 30 
# Set the confidence threshold for YOLO detection (Increased slightly to 0.10)
YOLO_CONFIDENCE_THRESHOLD = 0.10 
# Today's date (can be mocked for testing close calls)
TODAY = datetime.date.today() 

# --- IMPORTANT: UPDATE THIS PATH! ---
# Use the full, correct path to your best.pt file.
TRAINED_MODEL_PATH = 'best.pt' # EXAMPLE CORRECTION
# ---

# Global variables for models and OCR
MODEL = None
READER = None # Global variable to hold the EasyOCR reader instance

# --- Voice Alert Setup (Now using Console Print) ---

def generate_and_play_alert(text, is_error=False):
    """
    Prints alert text to the console, replacing the unreliable playsound feature 
    with a stable logging mechanism.
    """
    prefix = "VOICE ALERT (ERROR): " if is_error else "VOICE ALERT: "
    print(f"{prefix}{text}")


# --- Startup and Shutdown (Lifespan Context Manager) ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles application startup (loading models) and shutdown.
    """
    print("\n--- Starting FastAPI Server ---")
    
    # 1. Load YOLO Model
    try:
        print(f"Server: Attempting to load YOLO model from: {TRAINED_MODEL_PATH}")
        global MODEL
        MODEL = YOLO(TRAINED_MODEL_PATH)
        print("✅ YOLO Model loaded successfully.")
    except Exception as e:
        print(f"❌ Server Error: Failed to load YOLO model. Please check the TRAINED_MODEL_PATH defined in server.py.")
        print(f"Attempted Path: {TRAINED_MODEL_PATH}")
        print(f"Raw Error: {e}")
        generate_and_play_alert("Error loading Yolo model. Check server logs.", is_error=True)
        raise RuntimeError(f"Failed to load YOLO model from {TRAINED_MODEL_PATH}")
    
    # 2. EasyOCR Setup
    global READER
    if easyocr:
        try:
            # Use English language only for faster loading and date focus
            READER = easyocr.Reader(['en'])
            print("✅ OCR is now running using EasyOCR (English only).")
            generate_and_play_alert("EasyOCR model loaded.", is_error=False)
        except Exception as e:
            READER = None
            print(f"❌ EasyOCR Load Error: Failed to initialize reader. Error: {e}")
            print("⚠️ Falling back to MOCK OCR mode.")
            generate_and_play_alert("Error loading EasyOCR. Running in mock mode.", is_error=True)
    else:
        print("❌ EasyOCR Imports Failed: Module 'easyocr' not found.")
        print("⚠️ Falling back to MOCK OCR mode. Run 'pip install easyocr torch' for dynamic predictions.")


    print("✅ Audio alerts replaced with console logging for stability.")
    print(f"YOLO CONFIDENCE SET TO: {YOLO_CONFIDENCE_THRESHOLD}")
    print("INFO: Application startup complete. Waiting for requests...")
    generate_and_play_alert("Server startup complete.", is_error=False)

    yield
    
    # Shutdown logic
    print("INFO: Shutting down server...")

app = FastAPI(lifespan=lifespan)

# Allow CORS for front-end development
origins = [
    "http://localhost",
    "http://127.0.0.1",
    "http://127.0.0.1:8000",
    "http://127.0.0.1:8001",
    "http://127.00.1:5500", # Live Server
    "null", # Allow file:// for local file testing
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allowing all origins for simple local testing
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic model for request body
class ImageRequest(BaseModel):
    image_base64: str

# --- Image Processing Utilities ---

def base64_to_cv2(image_base64: str) -> np.ndarray:
    """Converts a base64 string to an OpenCV image (numpy array)."""
    try:
        img_bytes = base64.b64decode(image_base64)
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Could not decode image array.")
        return img
    except Exception as e:
        print(f"--- FATAL SERVER ERROR (base64_to_cv2) ---")
        print(e)
        raise HTTPException(status_code=400, detail="Invalid image base64 format.")

def enhance_contrast_clahe(image: np.ndarray) -> np.ndarray:
    """Applies Contrast Limited Adaptive Histogram Equalization (CLAHE) for enhancement."""
    try:
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        # Split the L, A, and B channels
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to the L-channel
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        
        # Merge the enhanced L channel back with the A and B channels
        limg = cv2.merge((cl, a, b))
        
        # Convert back to BGR color space
        final_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        return final_img
    except Exception as e:
        print(f"Contrast enhancement failed, returning original image. Error: {e}")
        return image


# --- OCR Implementation ---

def ocr_extraction_easyocr(cropped_image: np.ndarray) -> str:
    """
    Uses EasyOCR to extract text from the cropped image if the READER is available.
    Otherwise, falls back to the mock result.
    """
    global READER
    
    if READER is None:
        # Fallback to mock text if EasyOCR failed to load (e.g., ImportError)
        return "TEST-PRODUCT BATCH/001 DATE-05/26"

    try:
        # EasyOCR expects the image as a numpy array
        # result is a list of [bbox, text, confidence]
        results = READER.readtext(cropped_image) 
        
        # Join all detected text strings into a single string
        # Filter out low-confidence results (e.g., confidence < 0.4)
        extracted_text = " ".join([text for (bbox, text, conf) in results if conf > 0.4])
        return extracted_text.strip()
    except Exception as e:
        print(f"EasyOCR extraction failed: {e}")
        # Return an empty string on failure, which the caller handles as UNDETERMINED
        return ""

# --- Core Logic Functions ---

def parse_and_classify_date(extracted_text: str) -> dict:
    """
    Attempts to parse a date from a noisy text string and classifies it as EXPIRED, UNEXPIRED, 
    or UNDETERMINED. It uses the index-based closeness heuristic for refinement 
    and prioritizes specificity over lateness in the same year.
    """
    
    # Standardize spaces and make text uppercase for consistent matching
    text = extracted_text.upper().replace(' ', '').replace('.', '').replace(',', '')
    print(f"Normalized text for parsing: {text}")
    
    # Define Plausible Year Range to filter out OCR noise like '2500'
    PLANNED_MAX_YEAR = TODAY.year + 20 
    PLANNED_MIN_YEAR = 2000 

    # Check for keywords related to Manufacturing/Production vs. Expiration
    is_mfg_date = bool(re.search(r'(MFG|PROD|PRODUCTION|PACKED)', text))
    is_exp_date_explicit = bool(re.search(r'(EXP|EXPIRE|BEST|BB|USE BY)', text))


    # Define date patterns and their format codes (Ordered by specificity, most specific first)
    date_patterns = [
        # 1. Full Dates (DD/MM/YYYY or MM/DD/YYYY)
        (r'(\d{2})[/-](\d{2})[/-](\d{4})', ['%d/%m/%Y', '%m/%d/%Y']),
        
        # 2. MONTH/YEAR (M/YYYY or MM/YYYY) - e.g., '8/2027'
        (r'(\d{1,2})[/-](\d{4})', ['%m/%Y']), 
        
        # 3. Month Text/Year (e.g., MAR2025 or APR...2025)
        (r'(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)\D*(\d{4})', '%b%Y'), 
        (r'(\w{3})\s?(\d{4})', '%b%Y'),

        # 4. Month/Year (MM/YY)
        (r'(\d{2})[/-](\d{2})', ['%m/%y']),
        
        # 5. Year Only (Least specific, fallback. Use for month refinement.)
        (r'(\d{4})', '%Y'),
    ]

    best_date = None # Will store the latest successfully parsed and plausible date
    best_date_format = None # Track the format used to find the best date
    
    # Iterate through patterns to find ALL matches and keep the latest plausible date
    for pattern_group in date_patterns:
        pattern = pattern_group[0]
        formats_def = pattern_group[1] 

        # Ensure formats_to_try is always an iterable list
        formats_to_try = formats_def if isinstance(formats_def, list) else [formats_def]
        
        matches = re.findall(pattern, text)
        
        if not matches:
            continue

        for match in matches:
            # Determine the date string based on the match type
            date_str = "".join(match) if isinstance(match, tuple) else match

            # Try parsing with all associated formats
            for fmt in formats_to_try:
                current_parsed_date = None
                
                try:
                    
                    if fmt == '%m/%y':
                        # Handle MM/YY format (e.g., 03/27)
                        if isinstance(match, tuple) and len(match) == 2:
                            str_to_parse = f"{match[0]}/{match[1]}"
                        else:
                            str_to_parse = date_str
                        current_parsed_date = datetime.datetime.strptime(str_to_parse, '%m/%y').replace(day=1).date()
                    
                    elif fmt == '%m/%Y':
                        # Handles M/YYYY format (e.g., 8/2027, 10/2025)
                        month_str, year_str = match
                        str_to_parse = f"{month_str}/{year_str}"
                        current_parsed_date = datetime.datetime.strptime(str_to_parse, '%m/%Y').replace(day=1).date()

                    elif fmt == '%b%Y':
                        # Convert MONYYYY to the 1st of the month
                        current_parsed_date = datetime.datetime.strptime(date_str, fmt).replace(day=1).date()
                        
                    elif fmt == '%Y':
                        # Convert YYYY to the last day of the year (will be refined later if needed)
                        current_year = int(date_str)
                        current_parsed_date = datetime.date(current_year, 12, 31) 

                    else:
                        # Standard full date parsing (DD/MM/YYYY or MM/DD/YYYY)
                        current_parsed_date = datetime.datetime.strptime(date_str, fmt).date()
                        
                    
                    if current_parsed_date:
                        # PLAUSIBILITY CHECK: Filter out OCR noise like '2500'
                        if not (PLANNED_MIN_YEAR <= current_parsed_date.year <= PLANNED_MAX_YEAR):
                            print(f"Date {current_parsed_date.year} is implausible. Skipping.")
                            continue 

                        # --- FIX IMPLEMENTED HERE: Specificity over Lateness in the same year ---
                        is_current_specific = fmt != '%Y' # Is the current format M/Y, D/M/Y, or MONY?
                        is_best_specific = best_date_format is not None and best_date_format != '%Y'
                        
                        should_update = False

                        if best_date is None:
                            should_update = True
                        elif current_parsed_date.year > best_date.year:
                            # Always update if the year is strictly later
                            should_update = True
                        elif current_parsed_date.year == best_date.year:
                            if is_current_specific and not is_best_specific:
                                # Update: Current date is specific (e.g., 10/2025) and best is Y-only (e.g., 2025-12-31).
                                # The specific date is superior, so update.
                                should_update = True
                            elif (not is_current_specific and not is_best_specific) or (is_current_specific and is_best_specific):
                                # If both are Y-only, or both are specific (e.g., DD/MM/YYYY vs MM/YYYY), keep the later one
                                if current_parsed_date > best_date:
                                    should_update = True
                            # Else: Best date is already specific, and current is Y-only. Do not update.
                        
                        if should_update:
                            best_date = current_parsed_date
                            best_date_format = fmt # Store the format used
                            print(f"Updated best date: {best_date} from string '{date_str}' (Format: {fmt}) - Priority Update.")
                        
                        break 

                except ValueError:
                    continue 
            
    found_date = best_date 

    # --- MONTH REFINEMENT STEP (INDEX-BASED CLOSENESS) ---
    # Only refine if the best date was found using the Year Only pattern (%Y)
    if found_date and best_date_format == '%Y':
        print("INFO: Initiating Month Refinement (Best date was year-only).")
        month_pattern = r'(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)'
        
        # 1. Find all months and their spans (indices) in the normalized text
        all_month_matches = list(re.finditer(month_pattern, text))
        
        # 2. Find the latest year (as a string) and its span in the normalized text
        best_year_str = str(found_date.year)
        all_year_matches = list(re.finditer(best_year_str, text))
        
        if not all_year_matches:
             print(f"DEBUG: Could not find year string '{best_year_str}' in normalized text. Skipping refinement.")
        else:
            # We use the index of the latest instance of the year string found
            best_year_match = all_year_matches[-1]
            best_year_index = best_year_match.start()
            
            best_month_match = None
            min_distance = float('inf')

            # 3. Find the month closest to the latest year's position
            for month_match in all_month_matches:
                month_index = month_match.start()
                # Calculate absolute distance
                distance = abs(month_index - best_year_index)
                
                # If a closer month is found, update the best match
                if distance < min_distance:
                    min_distance = distance
                    best_month_match = month_match
            
            if best_month_match:
                latest_month_abbr = best_month_match.group(0)
                print(f"DEBUG: Closest month found to year {best_year_str} is '{latest_month_abbr}' (Distance: {min_distance}).")
                
                try:
                    # Get the month number
                    month_obj = datetime.datetime.strptime(latest_month_abbr, '%b') 
                    latest_month_num = month_obj.month

                    # Refine the date: use the best year and the closest month (day=1)
                    refined_date = found_date.replace(month=latest_month_num, day=1)
                    
                    print(f"Refined date: {refined_date.strftime('%d %m %Y')}")
                    found_date = refined_date
                    
                except ValueError:
                    print(f"DEBUG: Could not parse month abbreviation '{latest_month_abbr}' for refinement. Sticking to original date.")


    if not found_date:
        return {
            "status": "UNDETERMINED",
            "message": "Could not find or reliably parse any date in text. OCR may have failed or the date format is unsupported.",
            "classified_date": "N/A"
        }

    # --- Classification ---
    
    # Calculate difference based on the latest date found
    days_until_expiry = (found_date - TODAY).days
    
    # LOGIC ADJUSTMENT: Check for MFG keyword when the latest date is in the past
    if days_until_expiry < 0:
        if is_mfg_date and not is_exp_date_explicit:
            status = "MFG_DATE"
            message = f"**MANUFACTURING DATE DETECTED**. The date {found_date.strftime('%d %m %Y')} is likely the production date, not the expiration date. Cannot determine expiry."
            generate_and_play_alert(f"Manufacturing date detected: {found_date.strftime('%d %m %Y')}. Cannot determine expiry.")
            return {
                "status": status,
                "message": message,
                "classified_date": found_date.strftime('%d %m %Y')
            }
        
        # If no MFG/PROD keywords were found, treat it as an expired product
        status = "EXPIRED"
        message = f"Product is **EXPIRED**. The expiration date of {found_date.strftime('%d %m %Y')} was {abs(days_until_expiry)} days ago."
        generate_and_play_alert(f"Product is expired. The expiration date was {abs(days_until_expiry)} days ago.")

    elif days_until_expiry <= EXPIRATION_THRESHOLD_DAYS:
        status = "EXPIRING_SOON"
        message = f"Product is **EXPIRING SOON**! Only {days_until_expiry} days remaining (expires {found_date.strftime('%d %m %Y')})."
        generate_and_play_alert(f"Warning! Product is expiring soon. Only {days_until_expiry} days remaining.")
    else:
        status = "UNEXPIRED"
        message = f"Product is **UNEXPIRED**. Expiration date is {found_date.strftime('%d %m %Y')}."
        generate_and_play_alert(f"Product is unexpired. Expiration date is {found_date.strftime('%d %m %Y')}.")

    return {
        "status": status,
        "message": message,
        "classified_date": found_date.strftime('%d %m %Y')
    }

# --- API Endpoint ---

@app.post("/analyze_product")
async def analyze_product(request: ImageRequest):
    try:
        # 1. Base64 to CV2 Image (with contrast enhancement)
        cv_img_original = base64_to_cv2(request.image_base64)
        
        # 2. Pre-process for better YOLO detection
        cv_img_enhanced = enhance_contrast_clahe(cv_img_original)
        
        # 3. YOLO Detection
        # conf=YOLO_CONFIDENCE_THRESHOLD uses the global low threshold (0.10)
        results = MODEL.predict(cv_img_enhanced, conf=YOLO_CONFIDENCE_THRESHOLD, verbose=False)
        
        best_crop = None
        
        # Find the highest confidence 'expiry_date' box
        for result in results:
            # IMPORTANT: Check if any boxes were detected at all
            if not result.boxes or len(result.boxes) == 0:
                continue

            for box in result.boxes:
                # Get the class name using class index
                class_name = MODEL.names[int(box.cls)]
                
                if class_name == 'expiry_date':
                    # Extract coordinates
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    # Crop the image (using the original image, not the enhanced one)
                    # We add a small buffer (e.g., 5 pixels) for better OCR context
                    buffer = 5
                    x1 = max(0, x1 - buffer)
                    y1 = max(0, y1 - buffer)
                    x2 = min(cv_img_original.shape[1], x2 + buffer)
                    y2 = min(cv_img_original.shape[0], y2 + buffer)
                    
                    best_crop = cv_img_original[y1:y2, x1:x2]
                    
                    # We only process the first 'expiry_date' found for simplicity
                    break 
            if best_crop is not None:
                break


        if best_crop is None:
            # YOLO failed to detect the region
            generate_and_play_alert("No date region detected. Check model performance.")
            return {
                "status": "UNDETERMINED",
                "message": "No 'expiry_date' region was detected by the YOLO model. Try a clearer image, or the label is not visible.",
                "extracted_text": "N/A",
                "classified_date": "N/A"
            }

        # 4. Text Extraction using EasyOCR
        
        extracted_text = ocr_extraction_easyocr(best_crop)
        print(f"Raw Extracted Text: {extracted_text}")


        if not extracted_text or extracted_text == "TEST-PRODUCT BATCH/001 DATE-05/26":
            # If the extraction returns the mock text, it means EasyOCR failed to load.
            if extracted_text == "TEST-PRODUCT BATCH/001 DATE-05/26":
                message = "EasyOCR failed to load. Returning mock result. Please run 'pip install easyocr torch'."
                generate_and_play_alert("EasyOCR module is missing.", is_error=True)
            else:
                # OCR failed to extract any text
                message = "OCR failed to extract readable text from the detected region. Region may be blurry or obscured."
                generate_and_play_alert("OCR failed to extract readable text.")

            return {
                "status": "UNDETERMINED",
                "message": message,
                "extracted_text": extracted_text,
                "classified_date": "N/A"
            }

        # 5. Date Parsing and Classification
        classification_result = parse_and_classify_date(extracted_text)
        
        return {
            "status": classification_result["status"],
            "message": classification_result["message"],
            "extracted_text": extracted_text,
            "classified_date": classification_result["classified_date"]
        }

    except HTTPException as e:
        # Re-raise explicit HTTP exceptions
        raise e
    except Exception as e:
        # Handle all other unexpected server errors
        print(f"--- FATAL SERVER ERROR (analyze_product) ---")
        import traceback
        traceback.print_exc()
        generate_and_play_alert("Internal server error. Check server logs.")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {e}")

@app.get("/")
def read_root():
    return {"message": "Welcome to the Expiry Date Analyzer API. Go to /analyze_product to post an image."}

@app.get("/docs_info")
def docs_info():
    return {
        "docs_url": "/docs",
        "api_endpoint": "/analyze_product",
        "yolo_confidence": YOLO_CONFIDENCE_THRESHOLD,
        "expiration_threshold_days": EXPIRATION_THRESHOLD_DAYS,
        "today_date": TODAY.strftime('%Y-%m-%d')
    }
