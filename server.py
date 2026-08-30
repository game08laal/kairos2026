from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import io
from PIL import Image

app = FastAPI()

# Permite que seu Front-end HTML/JS converse com o Python sem bloqueios de CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Carrega a IA YOLOv8
modelo = YOLO("yolov8n.pt")

@app.post("/analisar")
async def analisar_imagem(file: UploadFile = File(...)):
    try:
        conteudo = await file.read()
        imagem = Image.open(io.BytesIO(conteudo))
        
        # Realiza a predição na imagem enviada pelo KAIROS
        resultados = modelo(imagem, conf=0.50)
        
        deteccoes = []
        for r in resultados:
            for box in r.boxes:
                classe_id = int(box.cls[0])
                nome_classe = modelo.names[classe_id]
                confianca = float(box.conf[0])
                
                deteccoes.append({
                    "classe": nome_classe,
                    "confianca": round(confianca * 100, 2)
                })
        
        return {"sucesso": True, "deteccoes": deteccoes}
    except Exception as e:
        return {"sucesso": False, "erro": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)