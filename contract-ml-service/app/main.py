from fastapi import FastAPI, HTTPException, Request, Depends, UploadFile, File, Header
from sqlalchemy.orm import Session
import json
import time
import uvicorn
import base64

from app.database import RequestHistory, get_db
from app.models import model

app = FastAPI(
    title="ML Service для классификации договоров",
    description="Сервис для классификации договоров по ОКПД2",
    version="1.0"
)

@app.post("/forward")
async def forward(
    request: Request,
    text: str = None,
    image: UploadFile = File(None),
    threshold: float = Header(0.2, description="Порог вероятности для предсказания"),
    db: Session = Depends(get_db)
):
    """
    Классификация договора.
    
    Варианты использования:
    1. multipart/form-data с изображением (параметр 'image')
    2. JSON с текстом (поле 'text')
    
    Дополнительные параметры в заголовках:
    - threshold: порог вероятности (по умолчанию 0.2)
    """
    start_time = time.time()
    
    try:
        # Вариант 1: Изображение через multipart/form-data
        if image:
            print(f"Получено изображение: {image.filename}, тип: {image.content_type}")
            
            # Проверяем формат изображения
            if image.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
                raise HTTPException(status_code=400, detail="bad request")
            
            # Читаем изображение
            image_data = await image.read()
            
            if len(image_data) == 0:
                raise HTTPException(status_code=400, detail="Изображение пустое")
            
            # Кодируем в base64 для возврата
            img_base64 = base64.b64encode(image_data).decode('utf-8')
            
            processing_time = time.time() - start_time
            
            # Сохраняем в базу
            history = RequestHistory(
                endpoint="/forward",
                request_data=json.dumps({
                    "type": "image",
                    "filename": image.filename,
                    "content_type": image.content_type,
                    "size_bytes": len(image_data),
                    "threshold": threshold
                }),
                response_data=json.dumps({
                    "image_processed": True,
                    "image_size": len(image_data),
                    "image_format": image.content_type
                }),
                processing_time=processing_time
            )
            db.add(history)
            db.commit()
            
            return {
                "type": "image",
                "image_processed": True,
                "image_size": len(image_data),
                "image_format": image.content_type,
                "image_base64_preview": img_base64[:100] + "..." if len(img_base64) > 100 else img_base64,
                "threshold": threshold,
                "processing_time": round(processing_time, 4)
            }
        
        # Вариант 2: Текст через JSON
        else:
            # Читаем JSON из тела запроса
            try:
                data = await request.json()
            except json.JSONDecodeError:
                raise HTTPException(status_code=400, detail="bad request")
            except Exception:
                raise HTTPException(status_code=400, detail="bad request")
            
            if "text" not in data:
                raise HTTPException(status_code=400, detail="bad request")
            
            text = data["text"].strip()
            
            if len(text) < 10:
                raise HTTPException(status_code=400, detail="Текст слишком короткий")
            
            # Используем модель с переданным порогом
            predictions = model.predict(text, threshold)
            
            if not predictions:
                raise HTTPException(status_code=403, detail="модель не смогла обработать данные")
            
            processing_time = time.time() - start_time
            
            # Сохраняем в базу
            history = RequestHistory(
                endpoint="/forward",
                request_data=json.dumps({
                    "type": "text", 
                    "text_preview": text[:100],
                    "threshold": threshold,
                    "text_length": len(text)
                }),
                response_data=json.dumps(predictions),
                processing_time=processing_time
            )
            db.add(history)
            db.commit()
            
            return {
                "type": "text",
                "text_preview": text[:100] + "..." if len(text) > 100 else text,
                "text_length": len(text),
                "predictions": predictions,
                "threshold": threshold,
                "processing_time": round(processing_time, 4)
            }
            
    except HTTPException:
        raise
    except Exception as e:
        print(f"Ошибка сервера: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/history")
async def get_history(db: Session = Depends(get_db)):
    """История запросов"""
    history = db.query(RequestHistory).all()
    
    result = []
    for item in history:
        try:
            request_data = json.loads(item.request_data) if item.request_data else {}
            response_data = json.loads(item.response_data) if item.response_data else {}
        except:
            request_data = {}
            response_data = {}
        
        result.append({
            "id": item.id,
            "endpoint": item.endpoint,
            "request": request_data,
            "response": response_data,
            "processing_time": item.processing_time,
            "timestamp": item.created_at.isoformat()
        })
    
    return result

@app.delete("/history")
async def delete_history(
    x_confirmation_token: str = Header(None, alias="x-confirmation-token"),
    db: Session = Depends(get_db)
):
    """Удаление истории запросов (требуется токен подтверждения в заголовке)"""
    
    if x_confirmation_token != "confirm-delete-123":
        raise HTTPException(
            status_code=400, 
            detail="Need confirmation token in header: x-confirmation-token: confirm-delete-123"
        )
    
    db.query(RequestHistory).delete()
    db.commit()
    
    return {"message": "История удалена"}

@app.get("/stats")
async def get_stats(db: Session = Depends(get_db)):
    """Статистика запросов"""
    history = db.query(RequestHistory).all()
    
    if not history:
        return {"message": "Нет данных для статистики"}
    
    times = [h.processing_time for h in history]
    text_lengths = []
    image_sizes = []
    
    # Анализ запросов
    for h in history:
        try:
            request_data = json.loads(h.request_data) if h.request_data else {}
            req_type = request_data.get('type', 'unknown')
            
            if req_type == 'text':
                length = request_data.get('text_length', 0)
                if length > 0:
                    text_lengths.append(length)
            elif req_type == 'image':
                size = request_data.get('size_bytes', 0)
                if size > 0:
                    image_sizes.append(size)
                    
        except:
            pass
    
    sorted_times = sorted(times)
    
    stats = {
        "total_requests": len(history),
        "request_types": {
            "text": len(text_lengths),
            "image": len(image_sizes),
            "unknown": len(history) - len(text_lengths) - len(image_sizes)
        },
        "processing_time": {
            "avg": round(sum(times) / len(times), 4),
            "p50": round(sorted_times[len(times)//2], 4),
            "p95": round(sorted_times[int(len(times)*0.95)] if len(times) > 1 else 0, 4),
            "p99": round(sorted_times[int(len(times)*0.99)] if len(times) > 1 else 0, 4)
        }
    }
    
    if text_lengths:
        stats["text_stats"] = {
            "avg_length": round(sum(text_lengths) / len(text_lengths), 1),
            "min_length": min(text_lengths),
            "max_length": max(text_lengths)
        }
    
    if image_sizes:
        stats["image_stats"] = {
            "avg_size_kb": round(sum(image_sizes) / len(image_sizes) / 1024, 1),
            "min_size_kb": round(min(image_sizes) / 1024, 1),
            "max_size_kb": round(max(image_sizes) / 1024, 1)
        }
    
    return stats

@app.get("/")
async def root():
    """Корневой эндпоинт"""
    return {
        "service": "ML Service для классификации договоров",
        "version": "1.0",
        "endpoints": {
            "POST /forward": {
                "description": "Классификация текста договора",
                "formats": [
                    "multipart/form-data с изображением (параметр 'image')",
                    "JSON с текстом (поле 'text')"
                ],
                "headers": {
                    "threshold": "Порог вероятности (по умолчанию 0.2)"
                }
            },
            "GET /history": "История запросов", 
            "DELETE /history": "Удаление истории (требует ?confirmation=yes)",
            "GET /stats": "Статистика запросов"
        }
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)