# Document-Question-Answering
An end-to-end OCR-free pipeline using Qwen3-VL that answers questions from both single-page and multi-page document images through prompt engineering and content-based page selection.


### For single Page docvqa file 
  specific_type_for_single_page.py
You can choose the type of document you want to run inference on by changing the `TARGET_TYPE` variable in the code.

```python
# ==============================
# CHANGE THIS
# ==============================
TARGET_TYPE = "layout"

# Available options:
# "handwritten"
# "table/list"
# "layout"
# "form"
# "figure/diagram"
# "Yes/No"
# ==============================



#For multipage docvqa
multipage_chunks.py

For the Multi-Page DocVQA dataset, inference can be executed in smaller chunks instead of processing the entire dataset at once.  
This is useful for:
- Reducing GPU/CPU memory usage
- Running experiments in parts
- Resuming inference from a specific point

You can control the range of samples processed by modifying:

```python
CHUNK_START = 0
CHUNK_END = 200
```

```python
# Process samples from index 0 to 199
CHUNK_START = 0
CHUNK_END = 200
```

```python
# Process samples from index 200 to 399
CHUNK_START = 200
CHUNK_END = 400
```

### Notes
- `CHUNK_START` → Starting sample index (inclusive)
- `CHUNK_END` → Ending sample index (exclusive)
- Adjust these values based on your system resources and dataset size.
- Multiple chunks can be processed separately and later combined if needed.
