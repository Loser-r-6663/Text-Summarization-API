import logging
from transformers import pipeline

# Thiết lập logging để theo dõi trạng thái tải model
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_NAME = "sshleifer/distilbart-cnn-12-6"

logger.info("Đang khởi tạo mô hình Hugging Face... Vui lòng đợi trong giây lát.")
try:
    summarizer = pipeline("summarization", model=MODEL_NAME)
    logger.info("Khởi tạo mô hình thành công!")
except Exception as e:
    logger.error(f"Lỗi khi tải mô hình: {e}")
    summarizer = None


def generate_summary(text: str, max_length: int = 130, min_length: int = 30) -> str:
    """
    Thực hiện tóm tắt văn bản truyền vào bằng mô hình Hugging Face.
    """
    if summarizer is None:
        raise RuntimeError("Mô hình chưa được khởi tạo thành công.")
    
    # Thực hiện suy luận (inference)
    result = summarizer(
        text, 
        max_length=max_length, 
        min_length=min_length, 
        do_sample=False
    )
    
    return result[0]["summary_text"]