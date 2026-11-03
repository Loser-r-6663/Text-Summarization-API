from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from model import generate_summary, MODEL_NAME

app = FastAPI(
    title="Text Summarization API",
    description="API tóm tắt văn bản sử dụng mô hình Hugging Face cho bài tập LAB-2",
    version="1.0.0"
)

# Cấu trúc dữ liệu đầu vào (Request Body) với Pydantic Validation
class SummarizeRequest(BaseModel):
    text: str = Field(..., min_length=50, description="Đoạn văn bản cần tóm tắt (tối thiểu 50 ký tự)")
    max_length: int = Field(default=130, ge=10, le=500, description="Độ dài tối đa của bản tóm tắt")
    min_length: int = Field(default=30, ge=5, le=200, description="Độ dài tối thiểu của bản tóm tắt")

# Cấu trúc dữ liệu đầu ra (Response Body)
class SummarizeResponse(BaseModel):
    model_used: str
    original_length: int
    summary_length: int
    summary_text: str


@app.get("/", summary="Giới thiệu hệ thống API")
def read_root():
    """
    Endpoint GET /: Trả về thông tin giới thiệu hệ thống API.
    """
    return {
        "system": "Text Summarization API",
        "description": "Dịch vụ tóm tắt văn bản tự động tích hợp mô hình Hugging Face DistilBART.",
        "version": "1.0.0",
        "endpoints": {
            "GET /": "Giới thiệu API",
            "GET /health": "Kiểm tra trạng thái hệ thống",
            "POST /predict": "Gửi văn bản và nhận kết quả tóm tắt"
        }
    }


@app.get("/health", summary="Kiểm tra trạng thái operational của hệ thống")
def health_check():
    """
    Endpoint GET /health: Kiểm tra tình trạng hoạt động của API.
    """
    return {
        "status": "healthy",
        "model_loaded": True
    }


@app.post("/predict", response_model=SummarizeResponse, summary="Tóm tắt văn bản")
def predict_summary(payload: SummarizeRequest):
    """
    Endpoint POST /predict: Nhận dữ liệu đầu vào, gọi mô hình AI và trả về kết quả JSON.
    """
    # Validation bổ sung: min_length không được lớn hơn max_length
    if payload.min_length >= payload.max_length:
        raise HTTPException(
            status_code=400, 
            detail="Tham số 'min_length' phải nhỏ hơn 'max_length'."
        )

    try:
        summary_result = generate_summary(
            text=payload.text,
            max_length=payload.max_length,
            min_length=payload.min_length
        )
        
        return SummarizeResponse(
            model_used=MODEL_NAME,
            original_length=len(payload.text),
            summary_length=len(summary_result),
            summary_text=summary_result
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Lỗi trong quá trình suy luận mô hình: {str(e)}"
        )