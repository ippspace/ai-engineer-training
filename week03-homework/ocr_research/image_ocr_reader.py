import logging
from enum import Enum
from pathlib import Path
from typing import Any, cast, override

from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document
from paddleocr import PaddleOCR
from paddlex.inference.pipelines.ocr.result import OCRResult

# 设置日志
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class OCRPredictStatus(Enum):
    """OCR预测状态枚举"""

    SUCCESS = "success"
    NO_TEXT_DETECTED = "no_text_detected"
    ERROR = "error"


class ImageOCRReader(BaseReader):
    """使用 PP-OCR v5 从图像中提取文本并返回 Document"""

    # 支持的图像格式
    SUPPORTED_FORMATS: set[str] = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".tiff",
        ".tif",
        ".webp",
        ".pdf",
    }

    def __init__(
        self, lang: str = "ch", use_gpu: bool = False, **kwargs: dict[str, Any]
    ):
        """
        Args:
            lang: OCR 语言 ('ch', 'en', 'fr', etc.)
            use_gpu: 是否使用 GPU 加速
            **kwargs: 其他传递给 PaddleOCR 的参数
        """
        device = "cpu"
        if use_gpu:
            device = "gpu"

        try:
            self.ocr_info: dict[str, str] = {
                "version": "PP-OCRv5",
                "lang": lang,
                "device": device,
            }
            self.ocr: PaddleOCR = PaddleOCR(
                ocr_version=self.ocr_info["version"],
                text_detection_model_name="PP-OCRv5_server_det",
                text_recognition_model_name="PP-OCRv5_server_rec",
                use_doc_orientation_classify=False,  # 通过 use_doc_orientation_classify 参数指定不使用文档方向分类模型
                use_doc_unwarping=False,  # 通过 use_doc_unwarping 参数指定不使用文本图像矫正模型
                use_textline_orientation=False,  # 通过 use_textline_orientation 参数指定不使用文本行方向分类模型
                lang=lang,
                device=device,
                **kwargs,
            )  # 更换 PP-OCRv5_server 模型
        except Exception as e:
            logger.error(f"初始化 PaddleOCR 失败: {e}")
            raise

    def _validate_file_paths(self, file_paths: list[str]) -> list[str]:
        """验证文件路径的有效性和格式支持"""
        valid_paths: list[str] = []

        for path in file_paths:
            path_obj = Path(path)

            # 检查文件是否存在
            if not path_obj.exists():
                logger.warning(f"文件路径 {path} 不存在，跳过处理")
                continue

            # 检查是否为文件（非目录）
            if not path_obj.is_file():
                logger.warning(f"路径 {path} 不是文件，跳过处理")
                continue

            # 检查文件格式
            file_suffix = path_obj.suffix.lower()
            if file_suffix not in self.SUPPORTED_FORMATS:
                logger.warning(f"文件 {path} 格式 {file_suffix} 不受支持，跳过处理")
                continue

            valid_paths.append(str(path_obj))

        return valid_paths

    def _extract_doc_from_image(self, image_path: str) -> Document | None:
        """从单个图像中提取文本"""

        try:
            logger.info(f"开始处理图像: {image_path}")

            result: list[OCRResult] = self.ocr.predict(input=image_path)
            if not result:
                logger.warning(f"图像 {image_path} 没有识别到文本")
                return None

            text_lines: list[str] = []
            all_scores: list[float] = []
            for res in result:
                # 从 res 对象中提取识别文本和置信度
                rec_texts = cast(list[str], res.get("rec_texts", []))
                rec_scores = cast(list[float], res.get("rec_scores", []))

                # 收集文本和分数
                text_lines.extend(rec_texts)
                all_scores.extend(rec_scores)

            # 拼接所有识别的文本
            full_text = "\n".join(text_lines)
            # 计算平均置信度
            avg_confidence = sum(all_scores) / len(all_scores) if all_scores else 0.0

            logger.info(f"成功处理图像 {image_path}，识别文本块数: {len(all_scores)}")

            metadata = {
                "image_path": image_path,
                "ocr_model": self.ocr_info["version"],
                "ocr_lang": self.ocr_info["lang"],
                "ocr_avg_confidence": round(avg_confidence, 4),
                "num_text_blocks": len(all_scores),
            }

            return Document(text=full_text, metadata=metadata)

        except Exception as e:
            logger.error(f"处理图像 {image_path} 时发生错误: {e}")
            return None

    @override
    def load_data(self, file: str | list[str]) -> list[Document]:
        """
        从单个或多个图像文件中提取文本，返回 Document 列表
        Args:
            file: 图像路径字符串 或 路径列表
        Returns:
            List[Document]
        """
        if not file:
            logger.warning("输入文件列表为空")
            return []

        # 统一转换为路径列表
        file_paths: list[str] = [file] if isinstance(file, str) else list(file)

        # 验证文件路径
        valid_paths = self._validate_file_paths(file_paths)

        if not valid_paths:
            logger.warning("没有找到有效的图像文件")
            return []

        logger.info(f"开始处理 {len(valid_paths)} 个图像文件")

        # 处理每个图像文件
        documents: list[Document] = []
        for path in valid_paths:
            document = self._extract_doc_from_image(path)
            if document is None:
                continue

            documents.append(document)

        logger.info(f"完成处理，共生成 {len(documents)} 个文档")
        return documents
