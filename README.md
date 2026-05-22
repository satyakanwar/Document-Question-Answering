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


