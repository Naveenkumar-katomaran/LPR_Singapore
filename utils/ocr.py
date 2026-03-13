import re
import logging as log

logging = log.getLogger('Rotating Log')

def get_max_string(cat_list):
    if not cat_list:
        return ""
    max_index = len(max(cat_list, key=len))
    max_string = ''
    positional_dict = {i: {} for i in range(max_index)}

    for text in cat_list:
        for i, char in enumerate(text):
            positional_dict[i][char] = positional_dict[i].get(char, 0) + 1

    for i in range(max_index):
        if positional_dict[i]:
            max_string += max(positional_dict[i], key=positional_dict[i].get)
    return max_string


def get_max(counted_list):
    max_value = None
    max_count = -1
    for item in counted_list:
        if item[1] > max_count:
            max_value = item[0]
            max_count = item[1]
    return [max_value, max_count]


# Indian Plate Mapping
CHAR_TO_NUM = {'B': '8', 'S': '5', 'G': '6', 'I': '1', 'O': '0', 'Z': '2', 'A': '4', 'T': '7'}
NUM_TO_CHAR = {v: k for k, v in CHAR_TO_NUM.items()}
# Add some extra common misreads
NUM_TO_CHAR.update({'0': 'D'}) # 0 and D are common
CHAR_TO_NUM.update({'D': '0'})

def apply_indian_corrections(plate):
    """
    Applies positional correction for Indian plates:
    Format: SS NN AA NNNN (or SS NN A NNNN)
    """
    plate = re.sub(r'[^A-Z0-9]', '', plate.upper())
    if len(plate) < 7:
        return plate
        
    chars = list(plate)
    
    # 1. State Code (0-1): Should be ALPHA
    for i in range(2):
        if chars[i].isdigit():
            chars[i] = NUM_TO_CHAR.get(chars[i], chars[i])
            
    # 2. RTO Code (2-3): Should be NUMERIC
    for i in range(2, 4):
        if chars[i].isalpha():
            chars[i] = CHAR_TO_NUM.get(chars[i], chars[i])
            
    # 3. Last 4 Digits: Should be NUMERIC
    # Identify the last 4 digits (usually indices -4 to -1)
    for i in range(len(chars) - 4, len(chars)):
        if chars[i].isalpha():
            chars[i] = CHAR_TO_NUM.get(chars[i], chars[i])
            
    # 4. Middle Letters (between RTO and Number): Should be ALPHA
    # Range: from index 4 to (len - 4)
    for i in range(4, len(chars) - 4):
        if chars[i].isdigit():
            chars[i] = NUM_TO_CHAR.get(chars[i], chars[i])
            
    return "".join(chars)

def is_valid_indian_plate(plate):
    """
    Validates against normal Indian plate pattern: ^[A-Z]{2}[0-9]{2}[A-Z]{1,2}[0-9]{4}$
    """
    # Normal format
    if re.match(r'^[A-Z]{2}[0-9]{2}[A-Z]{1,2}[0-9]{4}$', plate):
        return True
    # BH series: 22BH1234AA
    if re.match(r'^[0-9]{2}BH[0-9]{4}[A-Z]{2}$', plate):
        return True
    # Simplified check for variations
    if re.match(r'^[A-Z]{2}[0-9]{2}[A-Z0-9]{3,6}$', plate):
        return True
    return False

def consolidate_ocr_results(string_list, checksum_exclude):
    """
    Refined consolidation for Indian plates.
    1. Apply corrections to all strings.
    2. Filter for valid formats.
    3. Perform positional voting or frequency count.
    """
    if not string_list:
        return [None, 0], 0
        
    logging.debug(f"Input strings for consolidation: {string_list}")
    
    # Pre-process: remove noise and apply corrections
    cleaned_list = [re.sub(r'[^A-Z0-9]', '', s.upper()) for s in string_list]
    corrected_list = [apply_indian_corrections(s) for s in cleaned_list if len(s) >= 4]
    
    if not corrected_list:
        return [None, 0], 0

    # Separate valid formats from invalid ones
    valid_list = [s for s in corrected_list if is_valid_indian_plate(s)]
    
    target_list = valid_list if valid_list else corrected_list
    
    # Frequency based target
    target = get_max_string(target_list)
    
    # Final cleanup of results
    count = target_list.count(target)
    
    return [target, count], 1 if valid_list else 0
