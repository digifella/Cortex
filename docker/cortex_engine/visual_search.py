"""
Visual Search and Analysis Module
Enhanced image processing and search capabilities for Cortex Suite
"""

import os
import json
import base64
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from urllib.request import Request, urlopen
from llama_index.core import Document
from cortex_engine.config import VLM_MODEL
from cortex_engine.utils.logging_utils import get_logger
from cortex_engine.textifier import DocumentTextifier

logger = get_logger(__name__)

class VisualSearchEngine:
    """Enhanced visual processing and search engine"""
    
    def __init__(self, vlm_model: str = None):
        self.vlm_model = vlm_model or VLM_MODEL
        self._textifier = DocumentTextifier(use_vision=True)

    def _encode_image_for_vlm(self, image_path: str) -> str:
        with open(image_path, "rb") as image_file:
            prepared = self._textifier._prepare_vlm_image_bytes(image_file.read())
        return base64.b64encode(prepared).decode("utf-8")

    def _extract_response_text(self, response: Dict) -> str:
        message = (response or {}).get("message", {}) if isinstance(response, dict) else {}
        content = str((message or {}).get("content", "") or "").strip()
        thinking = str((message or {}).get("thinking", "") or "").strip()
        cleaned = self._textifier._normalize_vlm_text(content)
        if not cleaned and thinking:
            cleaned = self._textifier._normalize_vlm_text(thinking)
        return cleaned

    def _chat_vlm(self, prompt: str, encoded_images: List[str], num_predict: int) -> Dict:
        base_url = str(os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")).rstrip("/")
        url = f"{base_url}/api/chat"
        payload = {
            "model": self.vlm_model,
            "messages": [{
                "role": "user",
                "content": prompt,
                "images": encoded_images,
            }],
            "stream": False,
            "options": {
                "temperature": 0.1,
                "top_p": 0.9,
                "num_predict": num_predict,
            },
        }
        req = Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(req, timeout=180) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
        return json.loads(raw) if raw.strip() else {}
    
    def analyze_image_with_context(self, image_path: str, analysis_type: str = "comprehensive") -> Dict:
        """
        Analyze an image with specific focus based on analysis type
        
        Args:
            image_path: Path to the image file
            analysis_type: Type of analysis ("comprehensive", "ocr", "charts", "technical", "creative")
        
        Returns:
            Dictionary with structured analysis results
        """
        if not os.path.exists(image_path):
            return {"error": f"Image file not found: {image_path}"}
        
        try:
            encoded_image = self._encode_image_for_vlm(image_path)
            prompt = self._get_analysis_prompt(analysis_type)
            response = self._chat_vlm(
                prompt=prompt,
                encoded_images=[encoded_image],
                num_predict=800 if analysis_type == "comprehensive" else 400,
            )
            description = self._extract_response_text(response)
            
            analysis_result = {
                "image_path": image_path,
                "analysis_type": analysis_type,
                "description": description,
                "model_used": self.vlm_model,
                "success": True
            }
            
            # Extract structured data if possible
            if analysis_type == "ocr":
                analysis_result["extracted_text"] = self._extract_text_from_description(analysis_result["description"])
            elif analysis_type == "charts":
                analysis_result["chart_data"] = self._extract_chart_data(analysis_result["description"])
            
            return analysis_result
            
        except Exception as e:
            logger.error(f"Visual analysis failed for {image_path}: {e}")
            return {
                "error": str(e),
                "image_path": image_path,
                "analysis_type": analysis_type,
                "success": False
            }
    
    def _get_analysis_prompt(self, analysis_type: str) -> str:
        """Get specialized prompts for different analysis types"""
        
        prompts = {
            "comprehensive": '''Analyze this image comprehensively for a professional knowledge management system. Provide:

1. **Visual Content**: Describe what you see (objects, people, scenes, layout)
2. **Text Content**: Extract and transcribe any visible text, labels, titles, or captions
3. **Technical Elements**: Identify charts, graphs, diagrams, tables, or technical drawings
4. **Context & Purpose**: What is the likely purpose or message of this image?
5. **Key Information**: What are the most important details for knowledge retrieval?

Be specific and detailed. Include any data, numbers, or technical specifications visible.''',

            "ocr": '''Focus on extracting all visible text from this image. Please:

1. **Primary Text**: Transcribe all main text content (titles, headings, body text)
2. **Labels & Captions**: Extract any labels, captions, or annotations
3. **Data & Numbers**: Include any numerical data, measurements, or statistics
4. **Technical Text**: Include any technical terms, codes, or specifications
5. **Reading Order**: Present text in logical reading order

Prioritize accuracy over interpretation. If text is unclear, indicate with [unclear].''',

            "charts": '''Analyze any charts, graphs, or data visualizations in this image. Provide:

1. **Chart Type**: Identify the type of visualization (bar chart, pie chart, line graph, etc.)
2. **Data Points**: Extract specific data values, labels, and categories
3. **Trends & Patterns**: Describe any trends, patterns, or insights shown
4. **Axes & Scales**: Document axis labels, units, and scale information
5. **Key Findings**: What are the main conclusions from this data?

Focus on quantitative information and data accuracy.''',

            "technical": '''Analyze this image from a technical perspective. Include:

1. **Technical Components**: Identify any technical elements, systems, or equipment
2. **Specifications**: Extract any technical specifications, measurements, or parameters
3. **Diagrams & Schematics**: Describe technical diagrams, flowcharts, or schematics
4. **Process Steps**: Identify any procedural or process information
5. **Technical Context**: What technical domain or field does this relate to?

Focus on technical accuracy and professional terminology.''',

            "creative": '''Analyze this image with focus on creative and design elements:

1. **Visual Design**: Describe layout, composition, color scheme, and style
2. **Artistic Elements**: Identify artistic techniques, styles, or approaches used
3. **Brand Elements**: Note any logos, branding, or design patterns
4. **Creative Purpose**: What is the intended creative or marketing message?
5. **Aesthetic Impact**: How does the visual design support the content?

Focus on design principles and creative intent.'''
        }
        
        return prompts.get(analysis_type, prompts["comprehensive"])
    
    def _extract_text_from_description(self, description: str) -> List[str]:
        """Extract structured text elements from OCR analysis"""
        # Simple extraction - in production, this could be more sophisticated
        text_elements = []
        lines = description.split('\n')
        for line in lines:
            if line.strip() and not line.startswith(('1.', '2.', '3.', '4.', '5.')):
                text_elements.append(line.strip())
        return text_elements
    
    def _extract_chart_data(self, description: str) -> Dict:
        """Extract structured chart data from analysis"""
        # Placeholder for chart data extraction
        # In production, this could parse specific data points, values, etc.
        return {"raw_analysis": description}
    
    def compare_images(self, image1_path: str, image2_path: str) -> Dict:
        """Compare two images and identify similarities/differences"""
        try:
            encoded_img1 = self._encode_image_for_vlm(image1_path)
            encoded_img2 = self._encode_image_for_vlm(image2_path)
            prompt = '''Compare these two images and identify:

1. **Similarities**: What elements, content, or themes are similar between the images?
2. **Differences**: What are the key differences in content, style, or presentation?
3. **Context Relationship**: How might these images relate to each other in a document or process?
4. **Quality Assessment**: Are there differences in quality, resolution, or clarity?
5. **Recommendation**: Which image would be more suitable for different use cases?

Provide specific details about visual elements, text content, and overall presentation.'''
            response = self._chat_vlm(
                prompt=prompt,
                encoded_images=[encoded_img1, encoded_img2],
                num_predict=600,
            )
            
            return {
                "image1_path": image1_path,
                "image2_path": image2_path,
                "comparison": self._extract_response_text(response),
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Image comparison failed: {e}")
            return {"error": str(e), "success": False}
    
    def batch_analyze_images(self, image_paths: List[str], analysis_type: str = "comprehensive") -> List[Dict]:
        """Analyze multiple images in batch"""
        results = []
        
        for i, image_path in enumerate(image_paths, 1):
            logger.info(f"Analyzing image {i}/{len(image_paths)}: {Path(image_path).name}")
            result = self.analyze_image_with_context(image_path, analysis_type)
            results.append(result)
        
        return results
    
    def create_visual_summary(self, image_paths: List[str]) -> Dict:
        """Create a summary analysis of multiple related images"""
        if not image_paths:
            return {"error": "No images provided"}
        
        try:
            # Analyze each image briefly
            individual_analyses = []
            for path in image_paths:
                analysis = self.analyze_image_with_context(path, "comprehensive")
                if analysis.get("success"):
                    individual_analyses.append({
                        "path": path,
                        "summary": analysis["description"][:200] + "..." if len(analysis["description"]) > 200 else analysis["description"]
                    })
            
            # Create overall summary
            combined_summary = f"Visual collection analysis of {len(image_paths)} images:\n\n"
            for i, analysis in enumerate(individual_analyses, 1):
                image_name = Path(analysis["path"]).name
                combined_summary += f"{i}. **{image_name}**: {analysis['summary']}\n\n"
            
            return {
                "total_images": len(image_paths),
                "successful_analyses": len(individual_analyses),
                "combined_summary": combined_summary,
                "individual_analyses": individual_analyses,
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Visual summary creation failed: {e}")
            return {"error": str(e), "success": False}


# Convenience functions for easy integration
def analyze_image(image_path: str, analysis_type: str = "comprehensive") -> Dict:
    """Quick image analysis function"""
    engine = VisualSearchEngine()
    return engine.analyze_image_with_context(image_path, analysis_type)

def extract_text_from_image(image_path: str) -> Dict:
    """Extract text content from image using OCR analysis"""
    engine = VisualSearchEngine()
    return engine.analyze_image_with_context(image_path, "ocr")

def analyze_chart_image(image_path: str) -> Dict:
    """Analyze charts and data visualizations in image"""
    engine = VisualSearchEngine()
    return engine.analyze_image_with_context(image_path, "charts")
