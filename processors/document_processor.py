# processors/document_processor.py

import os
from typing import Dict, List, Tuple
from unstructured.partition.auto import partition
from utils.config import config
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentProcessor:
    def __init__(self):
        self.supported_formats = config.SUPPORTED_FORMATS

    async def process_document(self, file_path: str) -> List[Dict[str, any]]:
        """
        Process a document and extract its content with structure preservation.
        """
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
            
            file_ext = os.path.splitext(file_path)[1].lower()
            if file_ext not in self.supported_formats:
                raise ValueError(f"Unsupported file format: {file_ext}")
            
            # Extract content using unstructured-io
            elements = partition(filename=file_path)
            
            processed_elements = []
            position = 0
            
            for element in elements:
                element_type = type(element).__name__
                content = str(element)
                
                processed_element = {
                    "content": content,
                    "type": element_type,
                    "position": position,
                    "metadata": {
                        "source_file": file_path,
                        "element_type": element_type,
                    }
                }
                
                processed_elements.append(processed_element)
                position += len(content)
            
            return processed_elements

        except Exception as e:
            logger.error(f"Error processing document: {str(e)}")
            raise

    async def handle_complex_sections(self, elements: List[Dict]) -> List[Dict]:
        """
        Handle complex document sections using Gemini
        """
        # Implementation for complex section handling
        # This would integrate with Gemini for processing complex sections
        return elements